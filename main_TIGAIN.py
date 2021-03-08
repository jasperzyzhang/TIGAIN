'''Main function for the experiment include inputation method MICE,GAIN and TIGAIN
By Jasper Zhang
Date: 2021/03/08
Built based on GAIN by J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data
           Imputation using Generative Adversarial Nets," ICML, 2018.
           Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd

from data_loader import data_loader
from data_loader import data_loader_cols
from data_loader import data_loader_cols_TI
#from gain import gain
from TIgain import TIgain
from utils import rmse_loss
from fancyimpute import IterativeImputer
mice_impute = IterativeImputer()
from glob import glob


def main (args):
  '''Main function for UCI letter and spam datasets.
  
  Args:
    - data_name: GE or CN or ME
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    - col_num: the number of genes selected from data
    - result_name: the name of the result file
    - imp_type: imputation method name
    --info_name: name of extra info dataset(s)
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''



  result_name = args.result_name

  result_file = 'result/' + result_name + '.csv'

  if glob(result_file):
      print("result file exist")
      resultdf = pd.read_csv(result_file)
  else:
      print("result file DNE, create new file!")
      column_names = ["type", "colnum", "RMSE"]
      resultdf = pd.DataFrame(columns=column_names)

  data_name = args.data_name
  miss_rate = args.miss_rate
  col_num = args.col_num
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  
  # Load data and introduce missingness


  imp_type = args.imp_type

  if imp_type == 'mice' :      # Impute missing data using MICE
      ori_data_x, miss_data_x, data_m = data_loader_cols(data_name, miss_rate, col_num)

      mice_imputed_data_x = mice_impute.fit_transform(miss_data_x)
      # Report the RMSE performance
      mice_rmse = rmse_loss(ori_data_x, mice_imputed_data_x, data_m)

      print('MICE RMSE Performance: ' + str(np.round(mice_rmse, 4)))
      # Append result to the result file
      new_result_df = pd.DataFrame([["MICE", col_num, mice_rmse]],columns=['type', 'colnum', 'RMSE'])
      resultdf = resultdf.append(new_result_df, ignore_index=True)
      imputed_data_x = mice_imputed_data_x
      rmse = mice_rmse

  elif imp_type == 'gain' :   # Impute missing data using GAIN
      ori_data_x, miss_data_x, data_m = data_loader_cols(data_name, miss_rate, col_num)
      gain_imputed_data_x = gain(miss_data_x, gain_parameters)
      # Report the RMSE performance
      gain_rmse = rmse_loss(ori_data_x, gain_imputed_data_x, data_m)
      print('GAIN RMSE Performance: ' + str(np.round(gain_rmse, 4)))
      # Append result to the result file
      new_result_df = pd.DataFrame([ ["GAIN",col_num, gain_rmse]], columns=['type','colnum','RMSE'])
      resultdf = resultdf.append(new_result_df, ignore_index=True)

      imputed_data_x = gain_imputed_data_x
      rmse = gain_rmse

  elif imp_type == 'TIgain' :   # Impute missing data using TiGAIN
      print("Tightly Integrative GAIN")
      info_name = args.info_name
      ori_data_x, info_matrix , miss_data_x, data_m= data_loader_cols_TI(data_name,info_name, miss_rate, col_num)
      print("TIGAIN_load")
      gain_imputed_data_x = TIgain(miss_data_x,info_matrix, gain_parameters)

      # Report the RMSE performance
      gain_rmse = rmse_loss(ori_data_x, gain_imputed_data_x, data_m)
      print('GAIN RMSE Performance: ' + str(np.round(gain_rmse, 4)))
      # Append result to the result file
      ti_type = "TIGAN" + info_name
      new_result_df = pd.DataFrame([ [ti_type ,col_num, gain_rmse]], columns=['type','colnum','RMSE'])
      resultdf = resultdf.append(new_result_df, ignore_index=True)

      imputed_data_x = gain_imputed_data_x
      rmse = gain_rmse

  resultdf.to_csv(result_file, index=False, header=True)
  print("RMSE Result has been saved to " + result_file + ' .')


  return imputed_data_x, rmse

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['letter','spam','GE','ME','CN'],
      default='spam',
      type=str)
  parser.add_argument(
      '--miss_rate',
      help='missing data probability',
      default=0.2,
      type=float)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=128,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=10000,
      type=int)
  parser.add_argument(
      '--col_num',
      help='number of cols randomly selected from original dataset',
      default=10,
      type=int)
  parser.add_argument(
      '--result_name',
      help='result file name',
      default='newresult',
      type=str)
  parser.add_argument(
      '--imp_type',
      help='Imputation model name',
      choices=['mice', 'gain', 'TIgain'],
      default='gain',
      type=str)
  parser.add_argument(
      '--info_name',
      help='Extra_info_provide',
      choices=['GE','CN', 'ME', 'CON'],
      default='CN',
      type=str)
  
  args = parser.parse_args() 
  
  # Calls main function  
  imputed_data, rmse = main(args)
