"""Model class, to be extended by specific types of models. TBD"""
from pathlib import Path

import pandas as pd
import numpy as np
import pickle

DIRNAME = Path(__file__).parents[1].resolve() / 'weights'


Class Regressor:
	""" base class to be extended by other models """
	def __init__(self, model_fn: Callable, params: Dict = None):
		self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}'
		
		if params = None:
			params = {}
		self.model = model_fn(**params)
				
	def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
		"""
		Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
		Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
		"""
		maes = (y_true-y_pred).abs().groupby(types).mean()
		return np.log(maes.map(lambda x: max(x, floor))).mean()
		
	def fit(self, X, y):
		self.model.fit(X, y)
		
	def predict(self, X, y):
		return self.model.predict(X, y)
		
	def score(self, X, y):
		return self.model.score(X, y)
	
	def load_weights(self):
		self.model = pickle.load(open(filename), 'rb')
		#result = loaded_model.score(X_test, Y_test)
	
	def save_weights(self):
		filename = f'{self.name}_model.sav'
		pickle.dump(self.model, open(filename, 'wb')))