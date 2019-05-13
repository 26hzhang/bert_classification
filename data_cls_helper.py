import os
import csv
import collections
import tensorflow as tf
from bert import tokenization


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
                sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
                Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples to be a multiple of the batch size,
    because the TPU requires a fixed batch size. The alternative is to drop the last batch, which is bad because it
    means the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id, is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "msr_paraphrase_train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "msr_paraphrase_dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "msr_paraphrase_test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:  # remove header
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            text_b = tokenization.convert_to_unicode(line[4])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    @staticmethod
    def _read_tsv(input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class SickProcessor(DataProcessor):
    """Processor for the SICK data set."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "SICK_train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "SICK_trial.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "SICK_test_annotated.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["NEUTRAL", "ENTAILMENT", "CONTRADICTION"]

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            label = tokenization.convert_to_unicode(line[4])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    @staticmethod
    def _read_tsv(input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, mode="r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class SnliProcessor(DataProcessor):
    """Processor for the SNLI data set."""
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_data(data_dir, "train"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_data(data_dir, "dev"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_data(data_dir, "test"), "test")

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[2])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    @staticmethod
    def _read_data(input_file, file_name):
        # read sentence pair 1
        with open(input_file + "/s1." + file_name, mode="r", encoding="utf-8") as f:
            s1_list = []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                s1_list.append(line)
        # read sentence pair 2
        with open(input_file + "/s2." + file_name, mode="r", encoding="utf-8") as f:
            s2_list = []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                s2_list.append(line)
        # read label
        with open(input_file + "/labels." + file_name, mode="r", encoding="utf-8") as f:
            labels = []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                labels.append(line)

        assert len(s1_list) == len(s2_list) == len(labels), "the sentence pair and labels must be equal to each other!"
        lines = []
        for s1, s2, label in zip(s1_list, s2_list, labels):
            lines.append((s1, s2, label))
        return lines


class Sst2Processor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "sentiment-train")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "sentiment-dev")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "sentiment-test")), "test")

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[0])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples

    @staticmethod
    def _read_data(input_file):
        with open(input_file, mode="r", encoding="utf-8") as f:
            lines = []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                line = line.split("\t")
                if len(line) != 2:
                    continue
                lines.append(line)
        return lines


class Sst5Processor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "sentiment-train")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "sentiment-dev")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "sentiment-test")), "test")

    def get_labels(self):
        return ["0", "1", "2", "3", "4"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples

    @staticmethod
    def _read_data(input_file):
        with open(input_file, mode="r", encoding="utf-8") as f:
            lines = []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split(" ")
                label = tokens[0].strip()
                sentence = " ".join(tokens[1:])
                lines.append((label, sentence))
        return lines


class TrecProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "TREC.train.all")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "TREC.dev.all")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "TREC.test.all")), "test")

    def get_labels(self):
        return ["0", "1", "2", "3", "4", "5"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])

            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples

    @staticmethod
    def _read_data(input_file):
        with open(input_file, mode="r", encoding="utf-8") as f:
            lines = []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split(" ")
                label = tokens[0].strip()
                sentence = " ".join(tokens[1:])
                lines.append((label, sentence))
        return lines


class SubjProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])

            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples

    @staticmethod
    def _read_data(input_file):
        with open(input_file, mode="r", encoding="utf-8") as f:
            lines = []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split(" ")
                label = tokens[0].strip()
                sentence = " ".join(tokens[1:])
                lines.append((label, sentence))
        return lines


class MrProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples

    @staticmethod
    def _read_data(input_file):
        with open(input_file, mode="r", encoding="utf-8") as f:
            lines = []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split(" ")
                label = tokens[0].strip()
                sentence = " ".join(tokens[1:])
                lines.append((label, sentence))
        return lines


class CrProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples

    @staticmethod
    def _read_data(input_file):
        with open(input_file, mode="r", encoding="utf-8") as f:
            lines = []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split(" ")
                label = tokens[0].strip()
                sentence = " ".join(tokens[1:])
                lines.append((label, sentence))
        return lines


class ColaProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "in_domain_train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "in_domain_dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "out_of_domain_dev.tsv")), "test")

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if set_type == "test" and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    @staticmethod
    def _read_tsv(input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first sequence or the second sequence. The embedding
    # vectors for `type=0` and `type=1` were learned during pre-training and are added to the wordpiece embedding
    # vector (and position vector). This is not *strictly* necessary since the [SEP] token unambiguously separates the
    # sequences, but it makes it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is used as the "sentence vector". Note that
    # this only makes sense because the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 1:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % example.guid)
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature, tokens, label_id


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)
    batch_tokens, batch_labels = [], []
    for (ex_index, example) in enumerate(examples):

        feature, tokens, label_id = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)
        batch_tokens.append(tokens)
        batch_labels.append(label_id)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()
    return batch_tokens, batch_labels


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features_):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features_)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, dtype=tf.int32)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence one token at a time. This makes more
    # sense than truncating an equal percent of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        num_examples = len(features)
        # This is for demo purposes and does NOT scale to large data sets. We do not use Dataset.from_generator()
        # because that uses tf.py_func which is not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids": tf.constant(all_input_ids, shape=[num_examples, seq_length], dtype=tf.int32),
            "input_mask": tf.constant(all_input_mask, shape=[num_examples, seq_length], dtype=tf.int32),
            "segment_ids": tf.constant(all_segment_ids, shape=[num_examples, seq_length], dtype=tf.int32),
            "label_ids": tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):

        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)

        features.append(feature)
    return features
