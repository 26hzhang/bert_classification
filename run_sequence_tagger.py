import os
import pickle
import subprocess
import seq_metrics
import tensorflow as tf
from bert import modeling
from bert import optimization
from bert import tokenization
import data_seq_helper
from data_seq_helper import file_writer
from data_seq_helper import file_based_input_fn_builder
from data_seq_helper import filed_based_convert_examples_to_features

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("task_name", "ner", "The name of the task to train.")
flags.DEFINE_string("data_dir", "datasets/CoNLL2003_en", "The input data dir.")
flags.DEFINE_string("bert_config_file", "bert_ckpt/base_cased/bert_config.json", "The config json file")
flags.DEFINE_string("init_checkpoint", "bert_ckpt/base_cased/bert_model.ckpt", "Initial checkpoint")
flags.DEFINE_string("vocab_file", "bert_ckpt/base_cased/vocab.txt", "vocab file that the BERT model was trained on.")
flags.DEFINE_string("output_dir", "checkpoint/conll2003_en", "output dir where the model checkpoints will be written.")
flags.DEFINE_bool("do_lower_case", False, "Whether to lower case the input text.")
flags.DEFINE_integer("max_seq_length", 128, "The maximum total input sequence length after WordPiece tokenization.")
flags.DEFINE_bool("do_train", True, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", True, "Whether to run the model in inference mode on the test set.")
flags.DEFINE_integer("batch_size", 32, "Total batch size for training.")
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_float("num_train_epochs", 6.0, "Total number of training epochs to perform.")
flags.DEFINE_float("warmup_proportion", 0.1, "Proportion of training to perform linear learning rate warmup")
flags.DEFINE_integer("save_checkpoints_steps", 1000, "How often to save the model checkpoint.")
flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")
flags.DEFINE_bool("use_crf", True, "if use CRF for decode")


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels,
                 use_one_hot_embeddings, use_crf=False):
    """Creates a classification model."""
    model = modeling.BertModel(config=bert_config,
                               is_training=is_training,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               token_type_ids=segment_ids,
                               use_one_hot_embeddings=use_one_hot_embeddings)

    embedding = model.get_sequence_output()  # [batch_size, max_seq_length, hidden_size]

    max_seq_length = embedding.shape[-2].value
    hidden_size = embedding.shape[-1].value
    seq_len = tf.reduce_sum(tf.sign(tf.abs(input_ids)), reduction_indices=1)  # [batch_size]

    if is_training:
        embedding = tf.nn.dropout(embedding, keep_prob=0.9)

    embedding = tf.reshape(embedding, shape=[-1, hidden_size])  # [batch_size x max_seq_length, hidden_size]

    output_weights = tf.get_variable(name="output_weights",
                                     shape=[num_labels, hidden_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(name="output_bias",
                                  shape=[num_labels],
                                  initializer=tf.zeros_initializer())

    logits = tf.matmul(embedding, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, shape=[-1, max_seq_length, num_labels])

    if use_crf:
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(name="transition", shape=[num_labels, num_labels],
                                    initializer=modeling.create_initializer())
            log_likelihood, transition = tf.contrib.crf.crf_log_likelihood(inputs=logits,
                                                                           tag_indices=labels,
                                                                           transition_params=trans,
                                                                           sequence_lengths=seq_len)
            loss = tf.reduce_mean(-log_likelihood)
            predicts, viterbi_score = tf.contrib.crf.crf_decode(potentials=logits,
                                                                transition_params=transition,
                                                                sequence_length=seq_len)
            return loss, logits, predicts

    else:
        with tf.variable_scope("loss"):
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)  # [batch_sz, max_seq, num_labels]
            mask = tf.cast(input_mask, dtype=tf.float32)  # [batch_sz, max_seq]
            log_probabilities = tf.nn.log_softmax(logits, axis=-1)  # [batch_sz, max_seq, num_labels]
            per_sample_loss = -tf.reduce_sum(one_hot_labels * log_probabilities, axis=-1)  # [batch_sz, max_seq]
            loss = tf.reduce_sum(per_sample_loss * mask)
            predicts = tf.argmax(logits, axis=-1, output_type=tf.int32)
            return loss, logits, predicts


def model_fn_builder(bert_config, num_labels, init_ckpt, learning_rate, num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings, use_crf):

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, logits, predicts) = create_model(bert_config=bert_config,
                                                      is_training=is_training,
                                                      input_ids=input_ids,
                                                      input_mask=input_mask,
                                                      segment_ids=segment_ids,
                                                      labels=label_ids,
                                                      num_labels=num_labels,
                                                      use_one_hot_embeddings=use_one_hot_embeddings,
                                                      use_crf=use_crf)

        tvars = tf.trainable_variables()
        if init_ckpt:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                tvars, init_ckpt)
            tf.train.init_from_checkpoint(init_ckpt, assignment_map)

        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(loss=total_loss,
                                                     init_lr=learning_rate,
                                                     num_train_steps=num_train_steps,
                                                     num_warmup_steps=num_warmup_steps,
                                                     use_tpu=None)

            hook_dict = dict()
            hook_dict["loss"] = total_loss
            hook_dict["global_steps"] = tf.train.get_or_create_global_step()
            logging_hook = tf.train.LoggingTensorHook(hook_dict, every_n_iter=100)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                                          loss=total_loss,
                                                          train_op=train_op,
                                                          training_hooks=[logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(label_ids_, logits_, predicts_, mask_, num_labels_):
                predictions = tf.math.argmax(logits_, axis=-1, output_type=tf.int32)
                eval_loss = tf.metrics.mean_squared_error(labels=label_ids_, predictions=predicts_)
                cm = seq_metrics.streaming_confusion_matrix(labels=label_ids_,
                                                            predictions=predictions,
                                                            num_classes=num_labels_ - 1,
                                                            weights=mask_)
                return {
                    "eval_loss": eval_loss,
                    "confusion_matrix": cm
                }

            eval_metrics = (metric_fn, [label_ids, logits, predicts, input_mask, num_labels])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                                          loss=total_loss,
                                                          eval_metrics=eval_metrics)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                                          predictions=predicts)

        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "ner": data_seq_helper.NerProcessor,
        "chunk": data_seq_helper.ChunkProcessor,
        "pos": data_seq_helper.PosProcessor
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError("Cannot use sequence length %d because the BERT model was only trained up to sequence "
                         "length %d" % (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    if not os.path.exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()

    label_list = processor.get_labels(data_dir=FLAGS.data_dir)
    print(label_list, flush=True)

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    run_config = tf.contrib.tpu.RunConfig(
        cluster=None,
        master=None,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=8,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(data_dir=FLAGS.data_dir)
        num_train_steps = int(len(train_examples) / FLAGS.batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(bert_config=bert_config,
                                num_labels=len(label_list),
                                init_ckpt=FLAGS.init_checkpoint,
                                learning_rate=FLAGS.learning_rate,
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps,
                                use_one_hot_embeddings=False,
                                use_crf=FLAGS.use_crf)

    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(use_tpu=False,
                                            model_fn=model_fn,
                                            config=run_config,
                                            train_batch_size=FLAGS.batch_size,
                                            eval_batch_size=FLAGS.batch_size,
                                            predict_batch_size=FLAGS.batch_size)

    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num of Training examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        if not os.path.exists(train_file):
            filed_based_convert_examples_to_features(examples=train_examples,
                                                     label_list=label_list,
                                                     max_seq_length=FLAGS.max_seq_length,
                                                     tokenizer=tokenizer,
                                                     output_file=train_file,
                                                     output_dir=FLAGS.output_dir)

        train_input_fn = file_based_input_fn_builder(input_file=train_file,
                                                     seq_length=FLAGS.max_seq_length,
                                                     is_training=True,
                                                     drop_remainder=True)

        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(data_dir=FLAGS.data_dir)
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num of Evaluate examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        if not os.path.exists(eval_file):
            filed_based_convert_examples_to_features(examples=eval_examples,
                                                     label_list=label_list,
                                                     max_seq_length=FLAGS.max_seq_length,
                                                     tokenizer=tokenizer,
                                                     output_file=eval_file,
                                                     output_dir=FLAGS.output_dir)

        eval_input_fn = file_based_input_fn_builder(input_file=eval_file,
                                                    seq_length=FLAGS.max_seq_length,
                                                    is_training=False,
                                                    drop_remainder=False)

        result = estimator.evaluate(input_fn=eval_input_fn)
        tf.logging.info("***** Evaluation results *****")
        confusion_matrix = result["confusion_matrix"]
        p, r, f = seq_metrics.calculate(confusion_matrix, len(label_list) - 1)
        tf.logging.info("Precision = %s", str(p))
        tf.logging.info("Recall = %s", str(r))
        tf.logging.info("F1 = %s", str(f))

    if FLAGS.do_predict:
        with open(FLAGS.output_dir + '/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        predict_examples = processor.get_test_examples(data_dir=FLAGS.data_dir)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        batch_tokens, batch_labels = filed_based_convert_examples_to_features(examples=predict_examples,
                                                                              label_list=label_list,
                                                                              max_seq_length=FLAGS.max_seq_length,
                                                                              tokenizer=tokenizer,
                                                                              output_file=predict_file,
                                                                              output_dir=FLAGS.output_dir)
        tf.logging.info("***** Running prediction *****")
        tf.logging.info("  Num of Predicting examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

        predict_input_fn = file_based_input_fn_builder(input_file=predict_file,
                                                       seq_length=FLAGS.max_seq_length,
                                                       is_training=False,
                                                       drop_remainder=False)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")
        file_writer(output_predict_file=output_predict_file,
                    result=result,
                    batch_tokens=batch_tokens,
                    batch_labels=batch_labels,
                    id2label=id2label,
                    use_crf=FLAGS.use_crf)

        # run evaluation script
        subprocess.call("perl conlleval.pl -d '\t' < ./{}/label_test.txt".format(FLAGS.output_dir), shell=True)


if __name__ == "__main__":
    tf.app.run()
