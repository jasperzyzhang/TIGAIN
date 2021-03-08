'''Data loader for GE,CN,ME
by Jasper Zhang

Built based on GAIN by J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data
           Imputation using Generative Adversarial Nets," ICML, 2018.
           Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
'''

# Necessary packages
import numpy as np
from utils import binary_sampler
from keras.datasets import mnist


def data_loader (data_name, miss_rate):
  '''Loads datasets and introduce missingness.
  
  Args:
    - data_name: letter, spam, or mnist
    - miss_rate: the probability of missing components
    
  Returns:
    data_x: original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
  '''
  
  # Load data
  if data_name in ['letter', 'spam', 'GE']:
    file_name = 'data/'+data_name+'.csv'
    data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
  elif data_name == 'mnist':
    (data_x, _), _ = mnist.load_data()
    data_x = np.reshape(np.asarray(data_x), [60000, 28*28]).astype(float)

  # Parameters
  no, dim = data_x.shape
  
  # Introduce missing data
  data_m = binary_sampler(1-miss_rate, no, dim)
  miss_data_x = data_x.copy()
  miss_data_x[data_m == 0] = np.nan
      
  return data_x, miss_data_x, data_m

def simple_loader (data_name):
  '''Loads datasets by name

  Args:
    - data_name: letter, spam, or mnist
  Returns:
    data_x: original data

'''

  file_name = 'data/' + data_name + '.csv'

  data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
  print(str(data_name) + " Data Loaded!")

  return data_x


def cols_loader_colsample (data_x, col_num):
  '''Retrun randomly selected indicies of col_num columns from data_x

  Args:
    - data_x: data matrix
    - col_num: number of columns selected

  Returns:
     random_cols: randomly selected columns from data_x
  '''
  rown, coln = data_x.shape
  random_cols = np.random.choice(coln, size=col_num, replace=False)

  return random_cols


def data_loader_colsample (data_x, col_num):
  '''Retrun randomly selected columns from data_x

   Args:
    - data_x: data matrix
    - col_num: number of columns selected

  Returns:
    data_x: randomly selected columns from data_x
  '''
  # Parameters

  random_cols = cols_loader_colsample(data_x, col_num)

  data_sub = data_x[:, random_cols]

  data_x = data_sub

  return data_x

def data_loader_missing (data_x, miss_rate):
  ''' introduce missingness to data_x.

  Args:
    - data_x: input data matrix
    - miss_rate: the probability of missing components

  Returns:
    data_x: original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
  '''
  # Parameters
  no, dim = data_x.shape

  # Introduce missing data
  data_m = binary_sampler(1 - miss_rate, no, dim)
  miss_data_x = data_x.copy()
  miss_data_x[data_m == 0] = np.nan

  return data_x, miss_data_x, data_m

def data_loader_cols (data_name, miss_rate, col_num):
  ''' load data, randomly select columns of data and Introduce missingness.

  Args:
    - data_name: letter, spam, or mnist
    - miss_rate: the probability of missing components
    - col_num: number of columns selected

  Returns:
    data_x: selected cols from original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
  '''
  # Parameters

  data_x = simple_loader(data_name)

  data_x = data_loader_colsample (data_x, col_num)

  return data_loader_missing(data_x, miss_rate)



def data_loader_extrainfo(data_x, extra_info, miss_rate, col_num):
  ''' randomly select columns of data and extra info matrix and Introduce missingness to data_x.

  Args:
    - data_x: data matrix
    - extra_info: extra info matrix
    - miss_rate: the probability of missing components
    - col_num: number of columns selected

  Returns:
    data_x: selected cols from original data
    info_sub: selected cols from extra info matrix
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
  '''

  #randomly generated columns
  random_cols = cols_loader_colsample(data_x, col_num)

  data_sub = data_x[:, random_cols]

  info_sub = extra_info[:,random_cols]

  data_x, miss_data_x, data_m = data_loader_missing(data_sub, miss_rate)

  return data_x, info_sub, miss_data_x, data_m


def data_loader_cols_TI(data_name, info_name, miss_rate, col_num):
  ''' load data and extra info , randomly select columns of data and extra info and Introduce missingness to data.

  Args:
    - data_name: letter, spam, or mnist
    - info_name: name of extra info matrix GE, CN , ME or CON
    - miss_rate: the probability of missing components
    - col_num: number of columns selected

  Returns:
    data_x: selected cols from original data
    info_sub: selected cols from extra info matrix
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
  '''

  #load two data sets

  data_x = simple_loader(data_name)

  if info_name == 'CON': # use concatenation of two data as extra info
    if data_name == "GE":
      extra_info1 = simple_loader('CN')
      extra_info2 = simple_loader('ME')
    elif data_name == "ME":
      extra_info1 = simple_loader('CN')
      extra_info2 = simple_loader('GE')
    elif data_name == "CN":
      extra_info1 = simple_loader('GE')
      extra_info2 = simple_loader('ME')

    random_cols = cols_loader_colsample(data_x, col_num)

    data_sub = data_x[:, random_cols]

    info_sub1 = extra_info1[:, random_cols]
    info_sub2 = extra_info2[:, random_cols]

    info_sub = np.concatenate((info_sub1, info_sub2), axis=1)

    data_x, miss_data_x, data_m = data_loader_missing(data_sub, miss_rate)

    return data_x, info_sub, miss_data_x, data_m
  else:# use single data set as extra info
    extra_info = simple_loader(info_name)
    return data_loader_extrainfo(data_x, extra_info, miss_rate, col_num)


