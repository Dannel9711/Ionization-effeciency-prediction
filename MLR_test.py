import pandas as pd
import numpy as np

dataset = pd.read_csv(cosmo-1.csv)
dataset = dataset.dropna()

dataset.head()

X_train = dataset.iloc[:, 0:40].values
y_train = dataset.iloc[:, 40].values

dataset = pd.read_csv(cosmo-2.csv)
dataset = dataset.dropna()

dataset.head()

X_test = dataset.iloc[:, 0:40].values
y_test = dataset.iloc[:, 40].values


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn import linear_model

regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)
pred_test = model.predict(X_test)
print("Coefficients of sklearn: W=%s, b=%f" % (regressor.coef_, regressor.intercept_))

# Predictions for Training set
y_pred = regressor.predict(X_train).flatten()

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

from sklearn.metrics import r2_score

print(r2_score(y_train, y_pred))
import pandas as pd

df = pd.DataFrame({'predicted_values': y_pred, 'true_values': y_train, })
print(df)

df.to_csv('linear_tr.csv')


y_pred = regressor.predict(X_test).flatten()
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(r2_score(y_test, y_pred))
import pandas as pd

df = pd.DataFrame({'predicted_values': y_pred, 'true_values': y_test, })
print(df)
df.to_csv('linear_ts.csv')
