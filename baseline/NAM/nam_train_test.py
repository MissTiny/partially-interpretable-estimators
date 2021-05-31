# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Lint as: python3
"""Tests functionality of training NAM models."""

import os

from absl import flags
from absl import app
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import tensorflow.compat.v1 as tf

import nam_train

import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import KFold
import nam_train
from nam_train import FLAGS
import data_utils
import graph_builder
import functools
import sys
from absl import logging
FLAGS = flags.FLAGS


class NAMTrainingTest(parameterized.TestCase):
  """Tests whether NAMs can be run without error."""

  @parameterized.named_parameters(
      #('classification', 'BreastCancer', False),
      ('regression', 'Housing', True),
  )

  @flagsaver.flagsaver
  def test_nam(self, dataset_name, regression):
    """Test whether the NAM training pipeline runs successfully or not."""

    seed=1

    kf = KFold(n_splits = 5)
    
    FLAGS.save_checkpoint_every_n_epochs = 2
    FLAGS.early_stopping_epochs = 2
    FLAGS.dataset_name = 'CASP'

    logdir = os.path.join('CASP')
    tf.gfile.MakeDirs(logdir)
    df1 = pd.read_csv("Dataset/CASP_train.csv", delim_whitespace= False)
    df2 = pd.read_csv("Dataset/CASP_test.csv", delim_whitespace= False)
    df = pd.concat([df1,df2])
    col = df.columns
    y = df['Y']
    X = df.iloc[:,:-1]

    FLAGS.regression = FLAGS.reg
    logging.info(FLAGS.reg)
    logging.info(type(FLAGS.reg))

    FLAGS.shallow = FLAGS.shal
    FLAGS.training_epochs = FLAGS.epoch
    FLAGS.num_basis_functions = FLAGS.functions
    print("epoch:"+str(FLAGS.training_epochs)+" functions:"+str(FLAGS.num_basis_functions)+" regression"+str(FLAGS.regression)+" shallow:"+str(FLAGS.shallow))   
    fold=0
    all_results=[]
    for train_index,test_index in kf.split(X):
        fold+=1
        X_train,X_test =pd.DataFrame(X.to_numpy()[train_index], columns = col[0:-1]),pd.DataFrame(X.to_numpy()[test_index],columns=col[:-1])
        y_train,y_test =pd.DataFrame(y.to_numpy()[train_index],columns=['Y']),pd.DataFrame(y.to_numpy()[test_index],columns=['Y'])
        data_x, column_names = data_utils.transform_data(X_train)
        data_x = data_x.astype('float32')
                
        data_y = y_train.to_numpy().astype('float32')
        data_y = np.reshape(data_y,len(data_y))
        data_gen = data_utils.split_training_dataset(
            data_x,
            data_y,
            FLAGS.num_splits,
            stratified=not FLAGS.regression)#not True = False
        test_x, column_names = data_utils.transform_data(X_test)
        test_x = test_x.astype('float32')
        test_y = y_test.to_numpy().astype('float32')
        test_y = np.reshape(test_y,len(test_y))
        saver,graph_tensor, train, validation, real, train_op, test_op =nam_train.single_split_training(data_gen, test_x,test_y,logdir)
        #real=fold
        result = {'dataset':FLAGS.dataset_name,'fold':fold,'epoch':FLAGS.training_epochs,'basis':FLAGS.num_basis_functions,'test':real}
        all_results.append(result)
        csv_file="mouse_"+str(FLAGS.num_basis_functions)+".csv"
        with open(csv_file, 'w',newline ='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['dataset','fold','epoch','basis','test'])
            writer.writeheader()
            writer.writerow(result)


if __name__ == '__main__':
  #print(sys.argv[0])
  
  flags.DEFINE_integer('epoch', int(sys.argv[1]),
                     'The number of epochs to run training for.')
  flags.DEFINE_integer('functions', int(sys.argv[2]),
                     '')
  flags.DEFINE_boolean('reg', bool(sys.argv[3]),
                     '')
  flags.DEFINE_boolean('shal', bool(sys.argv[4]),
                     '')
  print(type(sys.argv[3]))
  print(bool(sys.argv[3]))
  del sys.argv[1]
  del sys.argv[1]
  del sys.argv[1]
  del sys.argv[1]
  #epoch=int(sys.argv[0])
  #functions = int(sys.argv[1])
  #regression = bool(sys.argv[2])
  #shallow = bool(sys.argv[3])
  #app.run(NAMTrainingTest.test_nam)
  #absltest.run_tests(argv=[sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3]],args=[],kwargs=[])
  absltest.main()

