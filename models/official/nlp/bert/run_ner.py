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
"""BERT classification finetuning runner in TF 2.x."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from absl import app
from absl import flags
from absl import logging

import numpy as np
# import custom_metrics
import pickle
import tensorflow as tf
import numpy as np

from official.modeling import model_training_utils
from official.nlp import optimization
from official.nlp.bert import bert_models
from official.nlp.bert import common_flags
from official.nlp.bert import configs as bert_configs
from official.nlp.bert import input_pipeline
from official.nlp.bert import model_saving_utils
from official.utils.misc import distribution_utils
#from official.utils.misc import keras_utils
# from official.nlp.bert import run_classifier
import tensorflow_hub as hub
from official.nlp.modeling.layers import CRF


flags.DEFINE_string('train_data_path', None,
                    'Path to training data for BERT classifier.')
flags.DEFINE_string('eval_data_path', None,
                    'Path to evaluation data for BERT classifier.')
flags.DEFINE_string('test_data_path', None,
                    'Path to testing data for BERT classifier.')
# Model training specific flags.
flags.DEFINE_string(
    'input_meta_data_path', None,
    'Path to file that contains meta data about input '
    'to be used for training and evaluation.')
flags.DEFINE_integer('train_batch_size', 32, 'Batch size for training.')
flags.DEFINE_integer('eval_batch_size', 32, 'Batch size for evaluation.')
flags.DEFINE_integer('test_batch_size', 32, 'Batch size for prediction.')

flags.DEFINE_string('save_history_path', None, 'Path to history file.')
flags.DEFINE_string('save_metric_path', None, 'Path to custom metric file.')
flags.DEFINE_boolean('is_training', True, 'if params is trainable')
flags.DEFINE_string('test_result_dir', None, 'test result dir')
flags.DEFINE_boolean("is_exporting_model", True, 'export model')
flags.DEFINE_integer('num_train_epoch', 10, 'epoch size for training.')

common_flags.define_common_bert_flags()

FLAGS = flags.FLAGS

def get_dataset_fn(input_file_pattern, max_seq_length, global_batch_size,
                   is_training):
  """Gets a closure to create a dataset."""

  def _dataset_fn(ctx=None):
    """Returns tf.data.Dataset for distributed BERT pretraining."""
    batch_size = ctx.get_per_replica_batch_size(
        global_batch_size) if ctx else global_batch_size
    dataset = input_pipeline.create_classifier_dataset(
        input_file_pattern,
        max_seq_length,
        batch_size,
        is_training=is_training,
        input_pipeline_context=ctx)
    return dataset

  return _dataset_fn

def acc_func(y_true, y_pred):
    batch_total_num = 0.0
    batch_correct_num = 0.0
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[1]):
            if (y_true[i][j] != 0 and y_true[i][j] != 4):
                batch_total_num += 1
                if(y_true[i][j] == y_pred[i][i]):
                    batch_correct_num += 1
    return batch_correct_num, batch_total_num


def compute_acc(ds, model):
    total_num = 0.0
    correct_num = 0.0
    for x, y in ds:
        batch_correct_num, batch_total_num = acc_func(y, model(x))
        total_num += batch_total_num
        correct_num += batch_correct_num
        break
    return correct_num / total_num


def BertCrf(model_file, max_length, num_labels, hidden_size=768):
    input_word_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_word_ids")
    segment_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="segment_ids")
    input_masks = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_masks")
    bert_layer = hub.KerasLayer(model_file, trainable=True)
    _, sequence_output = bert_layer([input_word_ids, input_masks, segment_ids])
    emission = tf.keras.layers.Dense(num_labels, activation="tanh")(sequence_output)
    crf = CRF.CRF(num_labels, name='crf_layer')
    decoded_sequence = crf(emission)
    model = tf.keras.Model(inputs=[input_word_ids, input_masks, segment_ids], outputs=decoded_sequence)
    model.compile(optimizer='adam', loss={'crf_layer': crf.get_loss})
    return model

def main(_):
  # Users should always run this script under TF 2.x
  assert tf.version.VERSION.startswith('2.')

  with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))

  if not FLAGS.model_dir:
    FLAGS.model_dir = '/tmp/bert20/'

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy,
      num_gpus=FLAGS.num_gpus,
      tpu_address=FLAGS.tpu)
  max_seq_length = input_meta_data['max_seq_length']
  train_input_fn = get_dataset_fn(
      FLAGS.train_data_path,
      max_seq_length,
      FLAGS.train_batch_size,
      is_training=True)
  eval_input_fn = get_dataset_fn(
      FLAGS.eval_data_path,
      max_seq_length,
      FLAGS.eval_batch_size,
      is_training=False)
  test_input_fn = get_dataset_fn(
      FLAGS.test_data_path,
      max_seq_length,
      FLAGS.test_batch_size,
      is_training=False)

  bert_config = bert_configs.BertConfig.from_json_file(FLAGS.bert_config_file)
  model = BertCrf(FLAGS.hub_module_url, max_length=max_seq_length, num_labels=input_meta_data["num_labels"] + 1)
  train_ds = train_input_fn()
  test_ds = test_input_fn()
  eval_ds = eval_input_fn()
  train_ds = train_ds.repeat(FLAGS.num_train_epoch)
  for step, (x,y) in enumerate(train_ds):
    model.fit(x, y)
    if step % 1 == 0:
        print("eval_acc of step%d: %.5f " % (step, compute_acc(eval_ds, model)))

  model.save(os.path.join(FLAGS.model_dir, "trained_model"))
  # add testing accuracy process
  print("test_acc  is  %.5f " % (compute_acc(test_ds, model)))

if __name__ == '__main__':
  flags.mark_flag_as_required('bert_config_file')
  flags.mark_flag_as_required('input_meta_data_path')
  flags.mark_flag_as_required('model_dir')
  app.run(main)



