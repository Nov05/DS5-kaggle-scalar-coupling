"""Define Basic Linear Regression"""

from sklearn.pipeline import Pipeline
import category_encoders as ce
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error


def linear(X, y):

	lr = LinearRegression()
	encoder = ce.OrdinalEncoder()
	#scaler = StandardScaler(with_mean=False)

	pipe = Pipeline(steps = [('encoder', encoder),
							 #('scaler', scaler),
							 ('lr', lr),
							])
	
	param_grid = {}
	
	search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=5)
	search.fit(X, y)
	print("Training Score (R^2): {}".format(search.best_score_))
	print("Best Parameters: {}".format(search.best_params_))
	print("Mean absolute error: {}".format(mean_absolute_error(y, search.best_estimator_.predict(X))))
	
	return search.best_estimator_