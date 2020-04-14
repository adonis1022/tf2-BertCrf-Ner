# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""BERT library to process data for classification task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import codecs
import pickle


from absl import logging
import tensorflow as tf

from official.nlp.bert import tokenization


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


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def __init__(self, process_text_fn=tokenization.convert_to_unicode):
    self.process_text_fn = process_text_fn

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

  @staticmethod
  def get_processor_name():
    """Gets the string identifier of the processor."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.io.gfile.GFile(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class NerDataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def __init__(self, process_text_fn=tokenization.convert_to_unicode, output_dir=None):
    self.process_text_fn = process_text_fn
    self.output_dir = output_dir
    self.labels = set()
    self.labels_from_data = set()

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

  @staticmethod
  def get_processor_name():
    """Gets the string identifier of the processor."""
    raise NotImplementedError()



class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self, process_text_fn=tokenization.convert_to_unicode):
    super(XnliProcessor, self).__init__(process_text_fn)
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "multinli",
                     "multinli.train.%s.tsv" % self.language))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = self.process_text_fn(line[0])
      text_b = self.process_text_fn(line[1])
      label = self.process_text_fn(line[2])
      if label == self.process_text_fn("contradictory"):
        label = self.process_text_fn("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
      language = self.process_text_fn(line[0])
      if language != self.process_text_fn(self.language):
        continue
      text_a = self.process_text_fn(line[6])
      text_b = self.process_text_fn(line[7])
      label = self.process_text_fn(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "XNLI"


class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "MNLI"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, self.process_text_fn(line[0]))
      text_a = self.process_text_fn(line[8])
      text_b = self.process_text_fn(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = self.process_text_fn(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "MRPC"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = self.process_text_fn(line[3])
      text_b = self.process_text_fn(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = self.process_text_fn(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "COLA"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if set_type == "test" and i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = self.process_text_fn(line[1])
        label = "0"
      else:
        text_a = self.process_text_fn(line[3])
        label = self.process_text_fn(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


class SstProcessor(DataProcessor):
  """Processor for the SST-2 data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "SST-2"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[0])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


class QnliProcessor(DataProcessor):
  """Processor for the QNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["entailment", "not_entailment"]

  @staticmethod
  def get_processor_name():
    """See base class."""
    return "QNLI"

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, 1)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        text_b = tokenization.convert_to_unicode(line[2])
        label = "entailment"
      else:
        text_a = tokenization.convert_to_unicode(line[1])
        text_b = tokenization.convert_to_unicode(line[2])
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class BdbkProcessor(DataProcessor):
    """Processor for bdbk data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_predict_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "predict.tsv")), "predict")

    def get_labels(self):
        """See base class."""
        return [
            'human',
            'poi',
            'publication',
            'organization',
            'webfic',
            'literature',
            'entertainment',
            'science',
            'other',
            'goods',
            'society',
            'food',
            'culture',
            'nature',
            'medical',
            'military'
        ]


class NerProcessor(NerDataProcessor):
    """Processor for Ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test"
        )

    def get_predict_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            # if i == 0:
            #     print('label: ', label)
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_labels(self, labels=None):
        # if labels is not None:
        #     try:
        #         # 支持从文件中读取标签类型
        #         if os.path.exists(labels) and os.path.isfile(labels):
        #             with codecs.open(labels, 'r', encoding='utf-8') as fd:
        #                 for line in fd:
        #                     self.labels.append(line.strip())
        #         else:
        #             # 否则通过传入的参数，按照逗号分割
        #             self.labels = labels.split(',')
        #         self.labels = set(self.labels) # to set
        #     except Exception as e:
        #         print(e)
        # # 通过读取train文件获取标签的方法会出现一定的风险。
        # if os.path.exists(os.path.join(self.output_dir, 'label_list.pkl')):
        #     with codecs.open(os.path.join(self.output_dir, 'label_list.pkl'), 'rb') as rf:
        #         self.labels = pickle.load(rf)
        # else:
        #     if len(self.labels) > 0:
        #         self.labels = self.labels.union(set(["X", "[CLS]", "[SEP]"]))
        #         with codecs.open(os.path.join(self.output_dir, 'label_list.pkl'), 'wb') as rf:
        #             pickle.dump(self.labels, rf)
        #     else:
        #         self.labels = ['B-PER', 'B-ORG', 'I-ORG', 'O', 'I-LOC', 'I-PER', 'B-LOC']
        return ['B-PER', 'B-ORG', 'I-ORG', 'O', 'I-LOC', 'I-PER', 'B-LOC', "X", "[CLS]", "[SEP]"]

    @staticmethod
    def get_processor_name():
        """See base class."""
        return "NER"

    def _read_data(self, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[-1])
                else:
                    if len(contends) == 0 and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                self.labels_from_data.add(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    continue
            return lines


# def convert_single_example(ex_index, example, label_list, max_seq_length,
#                            tokenizer):
#   """Converts a single `InputExample` into a single `InputFeatures`."""
#   label_map = {}
#   for (i, label) in enumerate(label_list):
#     label_map[label] = i
#
#   tokens_a = tokenizer.tokenize(example.text_a)
#   tokens_b = None
#   if example.text_b:
#     tokens_b = tokenizer.tokenize(example.text_b)
#
#   if tokens_b:
#     # Modifies `tokens_a` and `tokens_b` in place so that the total
#     # length is less than the specified length.
#     # Account for [CLS], [SEP], [SEP] with "- 3"
#     _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
#   else:
#     # Account for [CLS] and [SEP] with "- 2"
#     if len(tokens_a) > max_seq_length - 2:
#       tokens_a = tokens_a[0:(max_seq_length - 2)]
#
#   # The convention in BERT is:
#   # (a) For sequence pairs:
#   #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
#   #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
#   # (b) For single sequences:
#   #  tokens:   [CLS] the dog is hairy . [SEP]
#   #  type_ids: 0     0   0   0  0     0 0
#   #
#   # Where "type_ids" are used to indicate whether this is the first
#   # sequence or the second sequence. The embedding vectors for `type=0` and
#   # `type=1` were learned during pre-training and are added to the wordpiece
#   # embedding vector (and position vector). This is not *strictly* necessary
#   # since the [SEP] token unambiguously separates the sequences, but it makes
#   # it easier for the model to learn the concept of sequences.
#   #
#   # For classification tasks, the first vector (corresponding to [CLS]) is
#   # used as the "sentence vector". Note that this only makes sense because
#   # the entire model is fine-tuned.
#   tokens = []
#   segment_ids = []
#   tokens.append("[CLS]")
#   segment_ids.append(0)
#   for token in tokens_a:
#     tokens.append(token)
#     segment_ids.append(0)
#   tokens.append("[SEP]")
#   segment_ids.append(0)
#
#   if tokens_b:
#     for token in tokens_b:
#       tokens.append(token)
#       segment_ids.append(1)
#     tokens.append("[SEP]")
#     segment_ids.append(1)
#
#   input_ids = tokenizer.convert_tokens_to_ids(tokens)
#
#   # The mask has 1 for real tokens and 0 for padding tokens. Only real
#   # tokens are attended to.
#   input_mask = [1] * len(input_ids)
#
#   # Zero-pad up to the sequence length.
#   while len(input_ids) < max_seq_length:
#     input_ids.append(0)
#     input_mask.append(0)
#     segment_ids.append(0)
#
#   assert len(input_ids) == max_seq_length
#   assert len(input_mask) == max_seq_length
#   assert len(segment_ids) == max_seq_length
#
#   label_id = []
#   label_id.append(label_map["[CLS]"])
#   for i, token in enumerate(tokens):
#       tokens.append(token)
#       segment_ids.append(0)
#       label_id.append(label_map[label[i]])
#
#
#
#   if ex_index < 5:
#     logging.info("*** Example ***")
#     logging.info("guid: %s", (example.guid))
#     logging.info("tokens: %s",
#                  " ".join([tokenization.printable_text(x) for x in tokens]))
#     logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
#     logging.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
#     logging.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
#     logging.info("label: %s (id = %d)", example.label, label_id)
#
#   feature = InputFeatures(
#       input_ids=input_ids,
#       input_mask=input_mask,
#       segment_ids=segment_ids,
#       label_id=label_id,
#       is_real_example=True)
#   return feature

def convert_single_example1(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode=None):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param output_dir
    :param mode:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(output_dir+".dict", 'label2id.pkl')):
        os.makedirs(output_dir+".dict")
        with codecs.open(os.path.join(output_dir+".dict", 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    textlist = example.text_a.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），可以将所有的分字操作替换为list(input)
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:  # 一般不会出现else
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 打印部分样本数据信息
    if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        # logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_ids,
        is_real_example=True,
        # label_mask = label_mask
    )
    # mode='test'的时候才有效
    return feature

def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, output_dir=None, mode=None):
    """
    将数据转化为TF_Record 结构，作为模型数据输入
    :param examples:  样本
    :param label_list:标签list
    :param max_seq_length: 预先设定的最大序列长度
    :param tokenizer: tokenizer 对象
    :param output_file: tf.record 输出路径
    :param mode:
    :return:
    """
    writer = tf.io.TFRecordWriter(output_file)
    # 遍历训练数据
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        # 对于每一个训练样本,
        feature = convert_single_example1(ex_index, example, label_list, max_seq_length, tokenizer, output_file)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_id)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])
        # features["label_mask"] = create_int_feature(feature.label_mask)
        # tf.train.Example/Feature 是一种协议，方便序列化？？？
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()

def file_based_convert_examples_to_features(examples, label_list,
                                            max_seq_length, tokenizer,
                                            output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      logging.info("Writing example %d of %d", ex_index, len(examples))

    feature = convert_single_example1(ex_index, example, label_list,
                                     max_seq_length, tokenizer,output_dir=r"D:\work\bert-data")
    # feature = convert_single_example(ex_index, example, label_list,
                                      # max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def generate_tf_record_from_data_file(processor,
                                      data_dir,
                                      tokenizer,
                                      train_data_output_path=None,
                                      eval_data_output_path=None,
                                      test_data_output_path=None,
                                      max_seq_length=128):
  """Generates and saves training data into a tf record file.

  Arguments:
      processor: Input processor object to be used for generating data. Subclass
        of `DataProcessor`.
      data_dir: Directory that contains train/eval data to process. Data files
        should be in from "dev.tsv", "test.tsv", or "train.tsv".
      tokenizer: The tokenizer to be applied on the data.
      train_data_output_path: Output to which processed tf record for training
        will be saved.
      eval_data_output_path: Output to which processed tf record for evaluation
        will be saved.
      max_seq_length: Maximum sequence length of the to be generated
        training/eval data.

  Returns:
      A dictionary containing input meta data.
  """
  assert train_data_output_path or eval_data_output_path or test_data_output_path

  label_list = processor.get_labels()
  assert train_data_output_path
  train_input_data_examples = processor.get_train_examples(data_dir)
  filed_based_convert_examples_to_features(train_input_data_examples, label_list,
                                          max_seq_length, tokenizer,
                                          train_data_output_path)
  num_training_data = len(train_input_data_examples)

  if eval_data_output_path:
    eval_input_data_examples = processor.get_dev_examples(data_dir)
    filed_based_convert_examples_to_features(eval_input_data_examples,
                                            label_list, max_seq_length,
                                            tokenizer, eval_data_output_path)

  if test_data_output_path:
    test_input_data_examples = processor.get_test_examples(data_dir)
    filed_based_convert_examples_to_features(test_input_data_examples,
                                            label_list, max_seq_length,
                                            tokenizer, test_data_output_path)

  meta_data = {
      "task_type": "bert_classification",
      "processor_type": processor.get_processor_name(),
      "num_labels": len(processor.get_labels()),
      "train_data_size": num_training_data,
      "labels_list": processor.get_labels(),
      "max_seq_length": max_seq_length,
  }

  if eval_data_output_path:
    meta_data["eval_data_size"] = len(eval_input_data_examples)
  if test_data_output_path:
    meta_data["test_data_size"] = len(test_input_data_examples)
  return meta_data


def generate_predict_tf_record_from_data_file(processor,
                                              data_dir,
                                              tokenizer,
                                              predict_data_output_path=None,
                                              max_seq_length=128):
  assert predict_data_output_path

  label_list = processor.get_labels()
  predict_input_data_examples = processor.get_predict_examples(data_dir)
  file_based_convert_examples_to_features(predict_input_data_examples, label_list,
                                          max_seq_length, tokenizer,
                                          predict_data_output_path)
  meta_data = {
      "task_type": "bert_classification",
      "processor_type": processor.get_processor_name(),
      "num_labels": len(processor.get_labels()),
      "labels_list": processor.get_labels(),
      "max_seq_length": max_seq_length,
  }

  if predict_data_output_path:
    meta_data["predict_data_size"] = len(predict_input_data_examples)
  return meta_data
