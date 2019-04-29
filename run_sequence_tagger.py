import os
import pickle
import subprocess
import tensorflow as tf
from absl import logging
from argparse import ArgumentParser
import seq_metrics
from bert import modeling
from bert import optimization
from bert import tokenization
from data_seq_helper import file_writer
from data_seq_helper import NerProcessor
from data_seq_helper import PosProcessor
from data_seq_helper import ChunkProcessor
from data_seq_helper import boolean_string
from data_seq_helper import file_based_input_fn_builder
from data_seq_helper import filed_based_convert_examples_to_features

parser = ArgumentParser()
group1 = parser.add_argument_group("file path", "config the path, checkpoint and filename")
group1.add_argument("--task_name", type=str, default="ner", help="only accept [ner | chunk | pos]")
group1.add_argument("--data_dir", type=str, default="datasets/CoNLL2003_en", help="train, dev and test data dir")
group1.add_argument("--bert_path", type=str, default="bert_ckpt/base_uncased", help="pre-trained bert path")
group1.add_argument("--output_dir", type=str, default="checkpoint", help="output dir")

group2 = parser.add_argument_group("model config", "config the model parameters")
group2.add_argument("--gpu_idx", type=str, default="0", help="gpu idx")
group2.add_argument("--do_lower_case", type=boolean_string, default=True, help="whether to lowercase the input text")
group2.add_argument("--max_seq_length", type=int, default=128, help="maximal sequence length allowed")
group2.add_argument("--do_train", type=boolean_string, default=True, help="do training")
group2.add_argument("--do_eval", type=boolean_string, default=True, help="do evaluation")
group2.add_argument("--do_predict", type=boolean_string, default=True, help="do prediction")
group2.add_argument("--batch_size", type=int, default=16, help="training batch size")
group2.add_argument("--learning_rate", type=float, default=1e-4, help="initial learning rate for Adam")
group2.add_argument("--num_train_epochs", type=int, default=3, help="number of training epochs")
group2.add_argument("--warmup_proportion", type=float, default=0.1, help="learning rate warmup proportion")
group2.add_argument("--save_checkpoints_steps", type=int, default=1000, help="save checkpoints steps")
group2.add_argument("--save_summary_steps", type=int, default=1000, help="estimator call steps")
group2.add_argument("--add_rnn", type=boolean_string, default=False, help="use bidirectional rnn on top of bert model")
group2.add_argument("--use_crf", type=boolean_string, default=True, help="use CRF for output")

args = parser.parse_args()

bert_config_file = os.path.join(args.bert_path, "bert_config.json")
init_checkpoint = os.path.join(args.bert_path, "bert_model.ckpt")
vocab_file = os.path.join(args.bert_path, "vocab.txt")
output_dir = os.path.join(args.output_dir, args.data_dir.split("/")[-1])


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

    embedding = tf.reshape(embedding, shape=[-1, hidden_size])  # [batch_size x max_seq_length, hidden_size]

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

    if use_crf:
        logits = tf.reshape(logits, shape=[-1, max_seq_length, num_labels])
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
        return loss, logits, predicts, transition

    else:
        logits = tf.reshape(logits, shape=[-1, num_labels])
        labels = tf.reshape(labels, shape=[-1])
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        mask = tf.cast(input_mask, dtype=tf.float32)
        loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=one_hot_labels)
        loss = tf.reduce_mean(loss * mask) / (tf.reduce_mean(mask) + 1e-12)
        predicts = tf.argmax(tf.nn.softmax(logits, axis=-1), axis=-1)
        return loss, logits, predicts, None


def model_fn_builder(bert_config, num_labels, init_ckpt, learning_rate, num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings, use_crf):

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, logits, predicts, transition) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings, use_crf)

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
    # set GPUs and log level
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx
    logging.set_verbosity(logging.INFO)

    # load pre-trained bert configs
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError("The given seq len (%d) must smaller than or equal to the BERT max position embeddings (%d)" %
                         (args.max_seq_length, bert_config.max_position_embeddings))

    # create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processors = {"ner": NerProcessor, "chunk": ChunkProcessor, "pos": PosProcessor}

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()

    label_list = processor.get_labels(data_dir=args.data_dir)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=args.do_lower_case)

    sess_config = tf.ConfigProto(log_device_placement=False,
                                 inter_op_parallelism_threads=0,
                                 intra_op_parallelism_threads=0,
                                 allow_soft_placement=True)

    run_config = tf.estimator.RunConfig(model_dir=output_dir,
                                        save_checkpoints_steps=args.save_checkpoints_steps,
                                        session_config=sess_config)

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if args.do_train:
        train_examples = processor.get_train_examples(data_dir=args.data_dir)
        num_train_steps = int(len(train_examples) / args.batch_size * args.num_train_epochs)
        num_warmup_steps = int(num_train_steps * args.warmup_proportion)
        logging.info("***** Running training *****")
        logging.info("  Num of Training examples = %d", len(train_examples))
        logging.info("  Batch size = %d", args.batch_size)
        logging.info("  Num steps = %d", num_train_steps)

    model_fn = model_fn_builder(bert_config=bert_config,
                                num_labels=len(label_list),
                                init_ckpt=init_checkpoint,
                                learning_rate=args.learning_rate,
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps,
                                use_one_hot_embeddings=False,
                                use_crf=True)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       config=run_config,
                                       params={"batch_size": args.batch_size})

    if args.do_train:
        train_file = os.path.join(output_dir, "train.tf_record")
        if not os.path.exists(train_file):
            filed_based_convert_examples_to_features(examples=train_examples,
                                                     label_list=label_list,
                                                     max_seq_length=args.max_seq_length,
                                                     tokenizer=tokenizer,
                                                     output_file=train_file,
                                                     output_dir=output_dir)

        train_input_fn = file_based_input_fn_builder(input_file=train_file,
                                                     seq_length=args.max_seq_length,
                                                     is_training=True,
                                                     drop_remainder=True)

        early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(estimator=estimator,
                                                                            metric_name='loss',
                                                                            max_steps_without_decrease=num_train_steps,
                                                                            eval_dir=None,
                                                                            min_steps=0,
                                                                            run_every_secs=None,
                                                                            run_every_steps=args.save_checkpoints_steps)
        estimator.train(input_fn=train_input_fn,
                        max_steps=num_train_steps,
                        hooks=[early_stopping_hook])

    if args.do_eval:
        eval_examples = processor.get_dev_examples(data_dir=args.data_dir)
        logging.info("***** Running evaluation *****")
        logging.info("  Num of Evaluate examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", args.batch_size)

        eval_file = os.path.join(output_dir, "eval.tf_record")
        if not os.path.exists(eval_file):
            filed_based_convert_examples_to_features(examples=eval_examples,
                                                     label_list=label_list,
                                                     max_seq_length=args.max_seq_length,
                                                     tokenizer=tokenizer,
                                                     output_file=eval_file,
                                                     output_dir=output_dir)

        eval_input_fn = file_based_input_fn_builder(input_file=eval_file,
                                                    seq_length=args.max_seq_length,
                                                    is_training=False,
                                                    drop_remainder=False)
        result = estimator.evaluate(input_fn=eval_input_fn)
        logging.info("***** Evaluation results *****")
        confusion_matrix = result["confusion_matrix"]
        p, r, f = seq_metrics.calculate(confusion_matrix, len(label_list) - 1)
        logging.info("***********************************************")
        logging.info("***************Precision = %s******************", str(p))
        logging.info("***************Recall = %s   ******************", str(r))
        logging.info("***************F1 = %s       ******************", str(f))
        logging.info("***********************************************")

    if args.do_predict:
        with open(output_dir + '/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        predict_examples = processor.get_test_examples(args.data_dir)

        predict_file = os.path.join(output_dir, "predict.tf_record")
        batch_tokens, batch_labels = filed_based_convert_examples_to_features(examples=predict_examples,
                                                                              label_list=label_list,
                                                                              max_seq_length=args.max_seq_length,
                                                                              tokenizer=tokenizer,
                                                                              output_file=predict_file,
                                                                              output_dir=output_dir)
        logging.info("***** Running prediction *****")
        logging.info("  Num of Predicting examples = %d", len(predict_examples))
        logging.info("  Batch size = %d", args.batch_size)

        predict_input_fn = file_based_input_fn_builder(input_file=predict_file,
                                                       seq_length=args.max_seq_length,
                                                       is_training=False,
                                                       drop_remainder=False)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(output_dir, "label_test.txt")
        file_writer(output_predict_file=output_predict_file,
                    result=result,
                    batch_tokens=batch_tokens,
                    batch_labels=batch_labels,
                    id2label=id2label,
                    use_crf=args.use_crf)

        # run evaluation script
        subprocess.call("perl conlleval.pl -d '\t' < ./{}/label_test.txt".format(output_dir), shell=True)


if __name__ == "__main__":
    tf.app.run()
