import os
import codecs
import pickle
import collections
import tensorflow as tf
from bert import tokenization


class InputExample:
    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text: string. The untokenized text of the first sequence. For single
                sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures:
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class NerProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_example(self._read_data(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(self._read_data(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_example(self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self, data_dir=None):
        # recommend: get labels from train/dev/test files
        if data_dir is not None:
            label_counter = collections.Counter()
            files = ["train.txt", "dev.txt", "test.txt"]
            for file in files:
                with codecs.open(os.path.join(data_dir, file), mode="r", encoding="utf-8") as f:
                    for line in f:
                        contends = line.strip()
                        if len(contends) == 0:
                            continue
                        label = contends.split("\t")[-1].strip()
                        label_counter[label] += 1
            labels = ["[PAD]"] + [label for label, _ in label_counter.most_common()] + ["X", "[CLS]", "[SEP]"]
            return labels

        # default: CoNLL-2002/2003 NER labels (used only if you have the same datasets or the datasets hold the same
        # labels as follow)
        else:
            return ["[PAD]", "O", "B-LOC", "B-PER", "B-ORG", "I-PER", "I-ORG", "B-MISC", "I-LOC", "I-MISC", "X",
                    "[CLS]", "[SEP]"]

    @staticmethod
    def _create_example(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    @staticmethod
    def _read_data(input_file):
        """read BIO format"""
        with codecs.open(input_file, mode='r', encoding='utf-8') as f:
            lines, words, labels = [], [], []
            for line in f:
                contends = line.strip()
                tokens = contends.split("\t")
                if len(contends) == 0 and len(words) > 0:
                    words_, labels_ = [], []
                    for word, label in zip(words, labels):
                        if len(word) > 0 and len(label) > 0:
                            words_.append(word)
                            labels_.append(label)
                    lines.append([" ".join(labels_), " ".join(words_)])
                    words, labels = [], []
                else:
                    words.append(tokens[0])
                    labels.append(tokens[-1])
            return lines


class ChunkProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_example(self._read_data(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(self._read_data(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_example(self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self, data_dir=None):
        # recommend: get labels from train/dev/test files
        if data_dir is not None:
            label_counter = collections.Counter()
            files = ["train.txt", "dev.txt", "test.txt"]
            for file in files:
                with codecs.open(os.path.join(data_dir, file), mode="r", encoding="utf-8") as f:
                    for line in f:
                        contends = line.strip()
                        if len(contends) == 0:
                            continue
                        label = contends.split("\t")[-1].strip()
                        label_counter[label] += 1
            labels = ["[PAD]"] + [label for label, _ in label_counter.most_common()] + ["X", "[CLS]", "[SEP]"]
            return labels

        # default: CoNLL-2000 Chunk labels (used only if you have the same datasets or the datasets hold the same labels
        # as follow)
        else:
            return ["[PAD]", "I-NP", "B-NP", "O", "B-PP", "B-VP", "I-VP", "B-ADVP", "B-SBAR", "B-ADJP", "I-ADJP",
                    "B-PRT", "I-ADVP", "I-PP", "I-CONJP", "I-SBAR", "B-CONJP", "B-INTJ", "B-LST", "I-INTJ", "I-UCP",
                    "I-LST", "I-PRT", "B-UCP", "X", "[CLS]", "[SEP]"]

    @staticmethod
    def _create_example(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    @staticmethod
    def _read_data(input_file):
        """read BIO format"""
        with codecs.open(input_file, mode='r', encoding='utf-8') as f:
            lines, words, labels = [], [], []
            for line in f:
                contends = line.strip()
                tokens = contends.split("\t")
                if len(contends) == 0 and len(words) > 0:
                    words_, labels_ = [], []
                    for word, label in zip(words, labels):
                        if len(word) > 0 and len(label) > 0:
                            words_.append(word)
                            labels_.append(label)
                    lines.append([" ".join(labels_), " ".join(words_)])
                    words, labels = [], []
                else:
                    words.append(tokens[0])
                    labels.append(tokens[-1])
            return lines


class PosProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_example(self._read_data(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(self._read_data(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_example(self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self, data_dir=None):
        # recommend: get labels from train/dev/test files
        if data_dir is not None:
            label_counter = collections.Counter()
            files = ["train.txt", "dev.txt", "test.txt"]
            for file in files:
                with codecs.open(os.path.join(data_dir, file), mode="r", encoding="utf-8") as f:
                    for line in f:
                        contends = line.strip()
                        if len(contends) == 0:
                            continue
                        label = contends.split("\t")[-1].strip()
                        label_counter[label] += 1
            labels = ["[PAD]"] + [label for label, _ in label_counter.most_common()] + ["X", "[CLS]", "[SEP]"]
            return labels

        # default: Wall Street Journal (WSJ) POS labels (used only if you have the same datasets or the datasets hold
        # the same labels as follow)
        else:
            return ["[PAD]", "NN", "IN", "NNP", "DT", "JJ", "NNS", ",", ".", "CD", "RB", "VBD", "VB", "CC", "TO", "VBZ",
                    "VBN", "PRP", "VBG", "VBP", "MD", "POS", "PRP$", "``", "''", "$", ":", "WDT", "JJR", "NNPS", "WP",
                    "WRB", "JJS", "RBR", "RP", ")", "(", "EX", "RBS", "PDT", "FW", "WP$", "#", "UH", "SYM", "LS", "X",
                    "[CLS]", "[SEP]"]

    @staticmethod
    def _create_example(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    @staticmethod
    def _read_data(input_file):
        """read BIO format"""
        with codecs.open(input_file, mode='r', encoding='utf-8') as f:
            lines, words, labels = [], [], []
            for line in f:
                contends = line.strip()
                tokens = contends.split("\t")
                if len(contends) == 0 and len(words) > 0:
                    words_, labels_ = [], []
                    for word, label in zip(words, labels):
                        if len(word) > 0 and len(label) > 0:
                            words_.append(word)
                            labels_.append(label)
                    lines.append([" ".join(labels_), " ".join(words_)])
                    words, labels = [], []
                else:
                    words.append(tokens[0])
                    labels.append(tokens[-1])
            return lines


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir):
    """
    :param ex_index: example num
    :param example:
    :param label_list: all labels
    :param max_seq_length:
    :param tokenizer: WordPiece tokenization
    :param output_dir:
    :return: feature

    In this part we should rebuild input sentences to the following format.
        example:[Jim,Hen,##son,was,a,puppet,##eer]
        labels: [B-PER,I-PER,X,O,O,O,X]
    """
    label_map = {}
    # here start with zero this means that "[PAD]" is zero
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    with open(output_dir + "/label2id.pkl", 'wb') as w:
        pickle.dump(label_map, w)
    text_list = example.text.split(' ')
    label_list = example.label.split(' ')

    tokens, labels = [], []
    for i, (word, label) in enumerate(zip(text_list, label_list)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for m, _ in enumerate(token):
            if m == 0:
                labels.append(label)
            else:
                labels.append("X")

    # only Account for [CLS] with "- 1".
    if len(tokens) >= max_seq_length - 2:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]

    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])

    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])

    ntokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(label_map["[SEP]"])

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)

    # use zero to padding and you should
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("[PAD]")

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(ntokens) == max_seq_length

    # show processed examples
    if ex_index < 1:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % example.guid)
        tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_ids=label_ids)

    return feature, ntokens, label_ids


def filed_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file, output_dir):
    writer = tf.python_io.TFRecordWriter(output_file)

    batch_tokens, batch_labels = [], []
    for (ex_index, example) in enumerate(examples):
        feature, ntokens, label_ids = convert_single_example(ex_index=ex_index,
                                                             example=example,
                                                             label_list=label_list,
                                                             max_seq_length=max_seq_length,
                                                             tokenizer=tokenizer,
                                                             output_dir=output_dir)
        batch_tokens.extend(ntokens)
        batch_labels.extend(label_ids)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    writer.close()
    return batch_tokens, batch_labels


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features_):
        example = tf.parse_single_example(record, name_to_features_)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.data.experimental.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                                       batch_size=batch_size,
                                                       drop_remainder=drop_remainder))
        return d

    return input_fn


def _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i):
    token = batch_tokens[i]
    predict = id2label[prediction]
    true_l = id2label[batch_labels[i]]
    if token != "[PAD]" and token != "[CLS]" and true_l != "X" and token != "[SEP]":
        if predict == "X" and not predict.startswith("##"):
            predict = "O"
        line = "{}\t{}\t{}\n".format(token, predict, true_l)
        wf.write(line)


def file_writer(output_predict_file, result, batch_tokens, batch_labels, id2label):
    with open(output_predict_file, 'w') as wf:
        predictions = []
        for m, pred in enumerate(result):
            predictions.extend(pred)
        for i, prediction in enumerate(predictions):
            _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i)


def iob_to_iob2(labels):
    """Check that tags have a valid IOB format. Tags in IOB1 format are converted to IOB2."""
    for i, tag in enumerate(labels):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or labels[i - 1] == 'O':  # conversion IOB1 to IOB2
            labels[i] = 'B' + tag[1:]
        elif labels[i - 1][1:] == tag[1:]:
            continue
        else:
            labels[i] = 'B' + tag[1:]
    return True


def convert_iob_to_iobes(labels):
    """IOB -> IOBES"""
    iob_to_iob2(labels)
    new_tags = []
    for i, tag in enumerate(labels):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(labels) and labels[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(labels) and labels[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def convert_iob_to_iob2(labels):
    """Check that tags have a valid IOB format. Tags in IOB1 format are converted to IOB2."""
    new_tags = []
    for i, tag in enumerate(labels):
        if tag == 'O':
            new_tags.append(tag)
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            raise ValueError("Invalid NER tag: {}".format(tag))
        if split[0] == 'B':
            new_tags.append(tag)
        elif i == 0 or labels[i - 1] == 'O':  # conversion IOB1 to IOB2
            new_tags.append('B' + tag[1:])
        elif labels[i - 1][1:] == tag[1:]:
            new_tags.append(tag)
        else:
            new_tags.append('B' + tag[1:])
    return new_tags


def convert_iob2_or_iobes_to_iob(labels):
    new_tags = []
    for i, tag in enumerate(labels):
        if tag == "O":
            new_tags.append(tag)
        elif tag.startswith("B-"):
            new_tags.append("I" + tag[1:])
        elif tag.startswith("I-"):
            new_tags.append(tag)
        elif tag.startswith("S-"):
            new_tags.append("I" + tag[1:])
        else:
            raise ValueError("Invalid NER tag: {}".format(tag))
    return new_tags
