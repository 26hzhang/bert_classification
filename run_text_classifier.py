import os
import data_cls_helper
import tensorflow as tf
from bert import modeling
from bert import optimization
from bert import tokenization
from data_cls_helper import file_based_input_fn_builder
from data_cls_helper import file_based_convert_examples_to_features


flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("task_name", "mrpc", "The name of the task to train.")
flags.DEFINE_string("data_dir", "datasets/MRPC", "The input data dir.")
flags.DEFINE_string("bert_config_file", "bert_ckpt/base_uncased/bert_config.json", "The config json file")
flags.DEFINE_string("init_checkpoint", "bert_ckpt/base_uncased/bert_model.ckpt", "Initial checkpoint")
flags.DEFINE_string("vocab_file", "bert_ckpt/base_uncased/vocab.txt", "vocab file that the BERT model was trained on.")
flags.DEFINE_string("output_dir", "checkpoint/mrpc", "output directory where the model checkpoints will be written.")
flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the input text.")
flags.DEFINE_integer("max_seq_length", 128, "The maximum total input sequence length after WordPiece tokenization.")
flags.DEFINE_bool("do_train", True, "Whether to run training.")
flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", True, "Whether to run the model in inference mode on the test set.")
flags.DEFINE_integer("batch_size", 32, "Total batch size for training.")
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_float("num_train_epochs", 6.0, "Total number of training epochs to perform.")
flags.DEFINE_float("warmup_proportion", 0.1, "Proportion of training to perform linear learning rate warmup")
flags.DEFINE_integer("save_checkpoints_steps", 1000, "How often to save the model checkpoint.")
flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(config=bert_config,
                               is_training=is_training,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               token_type_ids=segment_ids,
                               use_one_hot_embeddings=use_one_hot_embeddings)

    embedding = model.get_pooled_output()

    hidden_size = embedding.shape[-1].value

    if is_training:
        embedding = tf.nn.dropout(embedding, keep_prob=0.9)

    output_weights = tf.get_variable(name="output_weights",
                                     shape=[num_labels, hidden_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(name="output_bias",
                                  shape=[num_labels],
                                  initializer=tf.zeros_initializer())

    logits = tf.matmul(embedding, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    predicts = tf.argmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return loss, per_example_loss, logits, probabilities, predicts


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate, num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings):

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities, predicts) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

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

            def metric_fn(per_example_loss_, label_ids_, logits_, is_real_example_):
                predictions = tf.argmax(logits_, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(labels=label_ids_, predictions=predictions, weights=is_real_example_)
                loss = tf.metrics.mean(values=per_example_loss_, weights=is_real_example_)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                                          loss=total_loss,
                                                          eval_metrics=eval_metrics)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                                          predictions={
                                                              "probabilities": probabilities,
                                                              "predictions": predicts
                                                          })

        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {"mrpc": data_cls_helper.MrpcProcessor,
                  "snli": data_cls_helper.SnliProcessor,
                  "sick": data_cls_helper.SickProcessor,
                  "cola": data_cls_helper.ColaProcessor,
                  "cr": data_cls_helper.CrProcessor,
                  "mr": data_cls_helper.MrProcessor,
                  "subj": data_cls_helper.SubjProcessor,
                  "sst5": data_cls_helper.Sst5Processor,
                  "sst2": data_cls_helper.Sst2Processor,
                  "trec": data_cls_helper.TrecProcessor}

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

    label_list = processor.get_labels()

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
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(len(train_examples) / FLAGS.batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(bert_config=bert_config,
                                num_labels=len(label_list),
                                init_checkpoint=FLAGS.init_checkpoint,
                                learning_rate=FLAGS.learning_rate,
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps,
                                use_one_hot_embeddings=False)

    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(use_tpu=False,
                                            model_fn=model_fn,
                                            config=run_config,
                                            train_batch_size=FLAGS.batch_size,
                                            eval_batch_size=FLAGS.batch_size,
                                            predict_batch_size=FLAGS.batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(input_file=train_file,
                                                     seq_length=FLAGS.max_seq_length,
                                                     is_training=True,
                                                     drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)", len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        eval_input_fn = file_based_input_fn_builder(input_file=eval_file,
                                                    seq_length=FLAGS.max_seq_length,
                                                    is_training=False,
                                                    drop_remainder=False)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        batch_tokens, batch_labels = file_based_convert_examples_to_features(
            predict_examples, label_list, FLAGS.max_seq_length, tokenizer, predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)", len(predict_examples),
                        num_actual_predict_examples, len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

        predict_input_fn = file_based_input_fn_builder(input_file=predict_file,
                                                       seq_length=FLAGS.max_seq_length,
                                                       is_training=False,
                                                       drop_remainder=False)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        total_examples, correct_predicts = 0, 0
        with tf.gfile.GFile(output_predict_file, mode="w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for i, (tokens, label, prediction) in enumerate(zip(batch_tokens, batch_labels, result)):
                probabilities = prediction["probabilities"]
                predict_label = prediction["predictions"]
                if i >= num_actual_predict_examples:
                    break
                total_examples += 1
                if predict_label == label:
                    correct_predicts += 1
                sentence = " ".join(tokens)
                class_probabilities = "\t".join(str(class_probability) for class_probability in probabilities)
                output_line = "\t".join([sentence, class_probabilities, str(label), str(predict_label)]) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples
        acc = float(correct_predicts) / float(total_examples)
        print("Test accuracy: {}".format(acc))


if __name__ == "__main__":
    tf.app.run()
