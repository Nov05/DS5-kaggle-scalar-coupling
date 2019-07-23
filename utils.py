import pandas as pd
import numpy as np
import psutil
import os


def reduce_mem_usage(df, verbose=True):
	"""
	This function reduces the numeric to the least possible numeric type that fits the data so 
	memory usage during transforming and training will be reduced.
	Taken from: https://www.kaggle.com/todnewman/keras-neural-net-for-champs

	Parameters:
	===========
	dataframe: input dataframe 
	verbose: verbose mode, default True.


	Output:
	===========
	dataframe: dataframe with numeric columns types changed to the least possible size

	"""

	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	start_mem = df.memory_usage().sum() / 1024**2    
	for col in df.columns:
		col_type = df[col].dtypes
		if col_type in numerics:
			c_min = df[col].min()
			c_max = df[col].max()
			if str(col_type)[:3] == 'int':
				if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
					df[col] = df[col].astype(np.int8)
				elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
					df[col] = df[col].astype(np.int16)
				elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
				elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
					df[col] = df[col].astype(np.int64)  
			else:
				if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
					df[col] = df[col].astype(np.float16)
				elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
					df[col] = df[col].astype(np.float32)
				else:
					df[col] = df[col].astype(np.float64)    
	end_mem = df.memory_usage().sum() / 1024**2
	if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
	return df


def show_ram_usage():
	"""
	This function shows current Python memory usage.
	"""
	py = psutil.Process(os.getpid())
	print('RAM usage: {} GB'.format(py.memory_info()[0]/2. ** 30))
	pass


def merge_train_structures(train, structures):
  
  """This function is used to merge the structures dataset to the 
     original train dataset
     
     Parameters:
     ===========
	   train: train dataframe
	   structures: structures dataframe.
     
     Output:
	   ===========
	   dataframe: merged dataframe
     """
  
  structures = structures.rename({'atom_index': 'atom_index_0',
                                  'x':'x_0', 'y':'y_0', 'z':'z_0',
                                  'atom':'atom_0'}, axis=1)
  
  merged = pd.merge(train, structures, on=['molecule_name', 'atom_index_0'])
  
  structures = structures.rename({'atom_index_0': 'atom_index_1',
                                  'x_0':'x_1', 'y_0':'y_1', 'z_0':'z_1',
                                  'atom_0':'atom_1'}, axis=1)
  
  merged_1 = pd.merge(merged, structures, on=['molecule_name', 'atom_index_1'])
  
  structures = structures.rename({'atom_index_1': 'atom_index',
                                  'x_1':'x', 'y_1':'y', 'z_1':'z',
                                  'atom_1':'atom'}, axis=1)
  
  assert train.shape[0] == merged.shape[0]
  
  return merged_1
