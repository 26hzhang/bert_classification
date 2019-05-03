# BERT Classification

Use google BERT (tensorflow-based) to do token-level and sentence-level classification.

## Requirements
- tensorflow>=1.11.0 (or tensorflow-gpu>=1.11.0)
- numpy>=1.14.4
- official tensorflow based bert code, get the code [`https://github.com/google-research/bert.git`](
https://github.com/google-research/bert.git) and place it under this repository.
- pre-trained bert models (according to the tasks), download and place to the `checkpoint/` directory.

```
bert_classification/
    |____ bert/
    |____ bert_ckpt/
    |____ checkpoint/
    |____ datasets/
    |____ .gitignore
    |____ conlleval.pl
    |____ data_cls_helper.py
    |____ data_seq_helper.py
    |____ README.md
    |____ run_sequence_tagger.py
    |____ run_text_classifier.py
```

## Dataset Overview

**Token level classification datasets (POS, Chunk and NER)**:

Dataset | Language | Classes | Training tokens | Dev tokens | Test tokens
:---: | :---: | :---: | :---: | :---: | :---:
CoNLL-2000 Chunk | English (en) | 23 | 211,727 | _N.A._ | 47,377
CoNLL-2002 NER | Spanish (es) | 9 | 207,484 (18,797) | 51,645 (4,351) | 52098 (3,558)
CoNLL-2002 NER | Dutch (nl) | 9 | 20,2931 (13,344) | 37,761 (2,616) | 68,994 (3,941)
CoNLL-2003 NER | English (en) | 9 | 20,4567 (23,499) | 51,578 (5,942) | 46,666 (5,648)
CoNLL-2003 NER | German (de) | 9 | - | - | -
Chinese NER 1 | Chinese (zh) | 21 | 1,044,967 (311,637) | 86,454 (24,444) | 119,467 (38,854)
Chinese NER 2 | Chinese (zh) | 7 | 979,180 (110,093) | 109,870 (12,059) | 219,197 (25,012)

> _CoNLL-2000 Chunk_ and _CoNLL-2002 NER_ datasets are obtained from [here](
https://github.com/teropa/nlp/tree/master/resources/corpora), _CoNLL-2003 English NER_ dataset is obtained from 
[here](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003), _CoNLL-2003 German NER_ dataset is obtained from 
[here](https://github.com/MaviccPRP/ger_ner_evals), Chinese NER 1 is obtained from [here](
https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset) and Chinese NER 2 is obtained from [here](
https://github.com/zjy-ucas/ChineseNER). All the lines in those datasets are convert to `(word, label)` pairs with `\t` 
as separator and drop all the `-DOCSTART-` lines.

**Sentence level classification datasets**:

Dataset | Classes | Average sentence length | Train size | Dev size | Test size
:---: | :---: | :---: | :---: | :---: | :---:
CR | 2 | 19 | 3,395 | _N.A._ | 377
MR | 2 | 20 | 9,595 | _N.A._ | 1,066
SST2 | 2 | 11 | 67,349 | 872 | 1,821
SST5 | 5 | 18 | 8,544 | 1,101 | 2,210
SUBJ | 2 | 23 | 9,000 | _N.A._ | 1,000
TREC | 6 | 10 | 5,452 | _N.A._ | 500

> All the datasets are converted to `utf-8` format. For the _SUBJ_, _MR_ and _CR_ datasets, `90%` for train, `10%` 
for test, while the dev dataset is the duplicate of test dataset. For _TREC_ dataset, the dev dataset is the duplicate 
of test dataset. Those datasets are obtained from [[facebookresearch/SentEval]](
https://github.com/facebookresearch/SentEval).

**Natural language inference (sentence pair classification) datasets**:

Dataset | Classes | Train size | Dev size | Test size
:---: | :---: | :---: | :---: | :---:
MRPC | 2 | 4,077 | 1,726 | 1,726
SICK | 3 | 4,501 | 501 | 4,928
SNLI | 3 | 549,367 | 9,842 | 9,824
CoLA | 2 | 8,551 | 527 | 516

> _MRPC_, _SICK_ and _SNLI_ are obtained from [[facebookresearch/SentEval]](
https://github.com/facebookresearch/SentEval), _CoLA_ us obtained from [[nyu-mll/GLUE-baselines]](
https://github.com/nyu-mll/GLUE-baselines). [_MNLI_](https://www.nyu.edu/projects/bowman/multinli/) and [_XNLI_](
https://www.nyu.edu/projects/bowman/xnli/) datasets are implemented by the official BERT already, see 
`run_classifier.py` in [[google-research/bert]](https://github.com/google-research/bert).

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

The token-level classification model contains two modules, one is using CRF for decode while another use a classifier 
directly. The output sequence of bert model is first fed into a dense layer and then decode by CRF/classifier, no 
intermediate RNN layers are used.

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
                               --num_train_epochs 6  # number of epochs
```

The sentence-level classification directly take the pooled output of bert model and feed it into a classifier for 
decode.

## Experiment Results

> All the experiments are running on `1` GeForce GTX 1080 Ti GPU.

**Token level classification datasets**

Dataset | en Chunk | es NER | nl NER | en NER | de NER | zh NER 1 | zh NER 2
:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:
Precision (%) | 96.8 | 89.0 | 89.8 | 92.0 | - | 77.9 | 95.7
Recall (%) | 96.4 | 88.6 | 90.0 | 90.8 | - | 73.1 | 95.7
F1 (%) | 96.6 | 88.8 | 89.9 | 91.4 | - | 75.5 | 95.7

> CoNLL-2002 Spanish/Dutch and CoNLL-2003 German NER use [`multi_cased_L-12_H-768_A-12.zip`](
https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) pre-trained model (base, 
multilingual, cased)  while CoNLL-2000 Chunk and CoNLL-2003 NER utilize [`cased_L-12_H-768_A-12.zip`](
https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip) pre-trained model (base, English, 
cased), Chinese NER uses [`chinese_L-12_H-768_A-12.zip`](
https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) pre-trained model (base, Chinese).

The testing results on CoNLL-2003 English NER are lower than the reported score of the [paper](
https://arxiv.org/pdf/1810.04805.pdf) (`91.4%` v.s. `92.4%`). As the paper says a `0.2%` difference is reasonable, 
however, I got `1.0%` error. I think maybe some tricks are missing, for example, the parameters setting in 
output classifier or data pre-processing strategies.

**Sentence level classification datasets**

Dataset | CR | MR | SST2 | SST5 | SUBJ | TREC
:---: | :---: | :---: | :---: | :---: | :---: | :---:
Dev Accuracy (%) | _N.A._ | _N.A._ | 91.3 | 50.1 | _N.A._ | _N.A._
Test Accuracy (%) | 89.2 | 85.4 | 93.5 | 53.3 | 97.3 | 96.6

> All the tasks use [`uncased_L-12_H-768_A-12.zip`](
https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) pre-trained model (base, English, 
uncased).

**Natural language inference datasets**

Dataset | MRPC | SICK | SNLI | CoLA
:---: | :---: | :---: | :---: | :---:
Dev Accuracy (%) | _N.A._ | 86.4 | 91.1 | 83.1
Test Accuracy (%) | 84.7 | 87.0 | 90.7 | 78.9

> All the tasks use [`uncased_L-12_H-768_A-12.zip`](
https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) pre-trained model (base, English, 
uncased). 

The results may differ from the reported results, since I do not use the _GLUE version_ datasets.

## Reference
- [[google-research/bert]](https://github.com/google-research/bert).
- [[macanv/BERT-BiLSTM-CRF-NER]](https://github.com/macanv/BERT-BiLSTM-CRF-NER).
- [[Kyubyong/bert_ner]](https://github.com/Kyubyong/bert_ner).
- [[kyzhouhzau/BERT-NER]](https://github.com/kyzhouhzau/BERT-NER).
- [[nyu-mll/GLUE-baselines]](https://github.com/nyu-mll/GLUE-baselines), the _MRPC_ data can be download [[here]](
https://github.com/jaisong87/prDetect/tree/master/Preprocess).
- [[facebookresearch/SentEval]](https://github.com/facebookresearch/SentEval).
- [[lancopku/Chinese-Literature-NER-RE-Dataset]](https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset), ref. 
[datasets/Chinese_NER_1](/datasets/Chinese_NER_1).
- [[zjy-ucas/ChineseNER]](https://github.com/zjy-ucas/ChineseNER), ref. [datasets/Chinese_NER_2](/datasets/Chinese_NER_2).
- [[MaviccPRP/ger_ner_evals]](https://github.com/MaviccPRP/ger_ner_evals), ref. CoNLL-2003 German NER dataset.
- [[teropa/nlp]](https://github.com/teropa/nlp), ref. CoNLL-2000 Chunk and CoNLL-2002 NER datasets.
- [[synalp/NER/corpus/CoNLL-2003]](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003), ref. CoNLL-2003 English 
NER dartaset.

