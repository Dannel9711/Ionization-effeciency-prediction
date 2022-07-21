import pandas as pd
import numpy as np
# fix random seed for reproducibility
np.random.seed(2022)


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
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(
    n_estimators=800,
    criterion='mse',
    max_depth=8,
    min_samples_split=11,
    min_samples_leaf=2,
    min_weight_fraction_leaf=0.0,
    max_features='sqrt',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None)

regressor.fit(X_train, y_train)
#Predictions for Training set
y_pred = regressor.predict(X_train).flatten()

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

import matplotlib.pyplot as plt
plt.scatter(y_train, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])
plt.show()


import pandas as pd
df = pd.DataFrame({'predicted_values':y_pred, 'true_values':y_train,})
print(df)

df.to_csv('forestplus_tr.csv')

from sklearn.metrics import r2_score
print(r2_score(y_train, y_pred))

#Predictions for testing set
y_pred = regressor.predict(X_test).flatten()

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

import pandas as pd
df = pd.DataFrame({'predicted_values':y_pred, 'true_values':y_test,})
print(df)

df.to_csv('forestplus_ts.csv')

print(r2_score(y_test, y_pred))
