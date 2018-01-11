import pandas as pd
import matplotlib.pyplot as plot
import sklearn

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

diabetes = load_diabetes()
diabetes_frame = pd.DataFrame(diabetes.data)
diabetes_frame.columns = diabetes.feature_names
diabetes_frame['Price'] = diabetes.target

X = diabetes_frame.drop('Price', axis=1)
lr = LinearRegression()
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, diabetes_frame.Price, test_size=0.15,
                                                                            random_state=5)
lr.fit(X_train, Y_train)
pred_train = lr.predict(X_train)
pred_test = lr.predict(X_test)

plot.scatter(pred_train, pred_train - Y_train, c='g', s=40, alpha=0.5)
plot.scatter(pred_test, pred_test - Y_test, c='y', s=40)
plot.hlines(y=0, xmin=0, xmax=50)
plot.title('Green points = train data, Yellow points = test data')
plot.ylabel('Residuals')
plot.show()
