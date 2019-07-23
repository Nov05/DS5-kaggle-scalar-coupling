"""Define Basic Linear Regression"""

from sklearn.pipeline import Pipeline
import category_encoders as ce
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


def Linear(X, y):

	lr = LinearRegression()
	encoder = ce.OrdinalEncoder()
	scaler = StandardScaler(with_mean=False)

	pipe = PipeLine(steps = [('encoder', encoder),
							 ('scaler', scaler),
							 ('lr', lr),
							])
	
	param_grid = {}
	
	search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=5)
	%time search.fit(X, y)
	print("Training Score (R^2): {}".format(search.best_score))
	print("Best Parameters: {}".format(search.best_params_))
	print("Training Accuracy: {}".format(search.best_estimator_predict(X, y)))
	
	return search.best_estimator_