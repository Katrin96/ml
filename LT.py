import pandas as pd
import matplotlib.pyplot as plot
import sklearn

from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso

boston = load_boston()
boston_frame = pd.DataFrame(boston.data)
boston_frame.columns = boston.feature_names
boston_frame['Price'] = boston.target

X = boston_frame.drop('Price', axis=1)
lt = Lasso()
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, boston_frame.Price, test_size=0.15,
                                                                            random_state=5)
lt.fit(X_train, Y_train)
pred_train = lt.predict(X_train)
pred_test = lt.predict(X_test)

plot.scatter(pred_train, pred_train - Y_train, c='g', s=40, alpha=0.5)
plot.scatter(pred_test, pred_test - Y_test, c='y', s=40)
plot.hlines(y=0, xmin=0, xmax=50)
plot.title('Green points = train data, Yellow points = test data')
plot.show()
