# BERT Classification

Use google BERT to do token-level and sentence-level classification.

## Requirements
- tensorflow>=1.11.0 (or tensorflow-gpu>=1.11.0)
- official tensorflow based bert code, get the code [`https://github.com/google-research/bert.git`](
https://github.com/google-research/bert.git) and place it under this repository.
- pre-trained bert models (according to the tasks), after downloading, place the model dir under `checkpoint/`.

```
bert_classification/
    |____ bert/
    |____ bert_ckpt/
    |____ checkpoint/
    |____ datasets/
    |____ conlleval.pl
    |____ data_cls_helper.py
    |____ ...
```

## Dataset Overview

**Token level classification datasets (POS, Chunk and NER)**, (`CoNLL` dataset):

Dataset | Language | Classes | Training tokens | Dev tokens | Test tokens
:---: | :---: | :---: | :---: | :---: | :---:
CoNLL-2000 Chunk | English (en) | 23 | 211727 | - | 47377
CoNLL-2002 NER | Spanish (es) | 9 | 207484 (18797) | 51645 (4351) | 52098 (3558)
CoNLL-2002 NER | Dutch (nl) | 9 | 202931 (13344) | 37761 (2616) | 68994 (3941)
CoNLL-2003 NER | English (en) | 9 | 204567 (23499) | 51578 (5942) | 46666 (5648)

> `CoNLL-2000 Chunking` and `CoNLL-2002 NER` datasets are obtained from [[teropa/nlp/resources/corpora]](
https://github.com/teropa/nlp/tree/master/resources/corpora), `CoNLL-2003 NER` dataset is obtained from 
[[synalp/NER/corpus/CoNLL-2003]](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003). All the lines in those 
datasets are convert to `(word, label)` pairs with `\t` as separator and drop all the `-DOCSTART-` lines.

**Sentence level classification datasets**, (`CR`, `MR`, `SST`, `SUBJ` and `TREC` datasets):

Dataset | Classes | Average sentence length | Train size | Dev size | Test size
:---: | :---: | :---: | :---: | :---: | :---:
CR | 2 | 19 | 3395 | - | 377
MR | 2 | 20 | 9595 | - | 1066
SST2 | 2 | 19 | 67349 | 872 | 1821
SST5 | 5 | 18 | 8544 | 1101 | 2210
SUBJ | 2 | 23 | 9000 | - | 1000
TREC | 6 | 10 | 5452 | - | 500

> All the datasets are converted to `utf-8` format. For the `SUBJ`, `MR` and `CR` datasets, `90%` for train, `10%` 
for test, while the dev dataset is the duplicate of test dataset. For `TREC` dataset, the dev dataset is the duplicate 
of test dataset. Those datasets are obtained from [[facebookresearch/SentEval]](
https://github.com/facebookresearch/SentEval).

**Natural language inference (sentence pair classification) datasets**, (`MRPC`, `SICK` and `SNLI` datasets):

Dataset | Classes | Train size | Dev size | Test size
:---: | :---: | :---: | :---: | :---:
MRPC | 2 | 4077 | 1726 | 1726
SICK | 3 | 4501 | 501 | 4928
SNLI | 3 | 549367 | 9842 | 9824

> Those datasets are obtained from [[facebookresearch/SentEval]](https://github.com/facebookresearch/SentEval). Note 
that [MNLI](https://www.nyu.edu/projects/bowman/multinli/) and [XNLI](https://www.nyu.edu/projects/bowman/xnli/) 
datasets are implemented by the official BERT already, see `run_classifier.py` in [[google-research/bert]](
https://github.com/google-research/bert).

## Usage
For token-level classification, run:
```bash
python3 run_sequence_tagger.py --task_name ner  \  # task name
                               --data_dir datasets/CoNLL2003_en  \  # dataset folder
                               --output_dir checkpoint/conll2003_en  \  # path to save outputs and trained params
                               --bert_config_file bert_ckpt/base_cased/bert_config.json  \  # pre-trained BERT configs
                               --init_checkpoint bert_ckpt/base_cased/bert_model.ckpt  \  # pre-trained BERT params
                               --vocab_file bert_ckpt/base_cased/vocab.txt  \  # BERT vocab file
                               --do_lower_case False  \  # whether lowercase the input tokens
                               --max_seq_length 128  \  # maximal sequence allowed
                               --do_train True  \  # if training
                               --do_eval True  \  # if evaluation
                               --do_predict True  \  # if prediction
                               --batch_size 32  \  # batch_size
                               --num_train_epochs 6  \  # number of epochs
                               --use_crf True  # if use CRF for decoding
```

For sentence-level classification, run:
```bash
python3 run_text_classifier.py --task_name mrpc  \  # task name
                               --data_dir datasets/MRPC  \  # dataset folder
                               --output_dir checkpoint/mrpc  \  # path to save outputs and trained params
                               --bert_config_file bert_ckpt/base_uncased/bert_config.json  \  # pre-trained BERT configs
                               --init_checkpoint bert_ckpt/base_uncased/bert_model.ckpt  \  # pre-trained BERT params
                               --vocab_file bert_ckpt/base_uncased/vocab.txt  \  # BERT vocab file
                               --do_lower_case True  \  # whether lowercase the input tokens
                               --max_seq_length 128  \  # maximal sequence allowed
                               --do_train True  \  # if training
                               --do_eval True  \  # if evaluation
                               --do_predict True  \  # if prediction
                               --batch_size 32  \  # batch_size
                               --num_train_epochs 6  \  # number of epochs
```

## Experiment Results

**Token level classification datasets**

Dataset | CoNLL-2000 en Chunk | CoNLL-2002 es NER | CoNLL-2002 nl NER | CoNLL-2003 en NER
:---: | :---: | :---: | :---: | :---:
Precision (%) | - | - | - | -
Recall (%) | - | - | - | -
F1 (%) | - | - | - | -


**Sentence level classification datasets**

Dataset | CR | MR | SST2 | SST5 | SUBJ | TREC
:---: | :---: | :---: | :---: | :---: | :---: | :---:
Dev Accuracy (%) | - | - | 91.3 | 50.1 | - | -
Test Accuracy (%) | 89.2 | 85.4 | 93.5 | 53.3 | 97.3 | 96.6

**Natural language inference datasets**

Dataset | MRPC | SICK | SNLI
:---: | :---: | :---: | :---:
Dev Accuracy (%) | - | - | -
Test Accuracy (%) | - | - | -

## Reference
- [[google-research/bert]](https://github.com/google-research/bert).
- [[macanv/BERT-BiLSTM-CRF-NER]](https://github.com/macanv/BERT-BiLSTM-CRF-NER).
- [[Kyubyong/bert_ner]](https://github.com/Kyubyong/bert_ner).
- [[kyzhouhzau/BERT-NER]](https://github.com/kyzhouhzau/BERT-NER).
