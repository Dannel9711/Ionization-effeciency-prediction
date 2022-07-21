from __future__ import absolute_import, division, print_function
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import numpy as np
np.random.seed(2022)
print(tf.__version__)

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



# Compile the ANN model
def build_model():
  model = keras.Sequential([
    keras.layers.Flatten(input_shape=(X_train.shape[1],)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(100, activation='exponential'),
    keras.layers.Dense(1)
  ])

  optimizer = tf.optimizers.Adamax(0.001)

  model.compile(loss='mae',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 20

# Store training stats
history = model.fit(X_train, y_train, epochs=EPOCHS,
                     verbose=0,
                    callbacks=[PrintDot()])

y_pred = model.predict(X_train).flatten()
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

from sklearn.metrics import r2_score

print(r2_score(y_train, y_pred))

df = pd.DataFrame({'predicted_values': y_pred, 'true_values': y_train, })
print(df)

df.to_csv('ann_tr.csv')

y_pred = model.predict(X_test).flatten()
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(r2_score(y_test, y_pred))
import pandas as pd

df = pd.DataFrame({'predicted_values': y_pred, 'true_values': y_test, })
print(df)
df.to_csv('ann_ts.csv')