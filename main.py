import numpy as np
import matplotlib.pyplot as plt

import importlib

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

import sklearn.linear_model as lm

import mp1
importlib.reload(mp1)





# Generate train set
[X_train, Y_train] = mp1.generate_dataset_classification(300, 20)

Y_train_onehot = (np_utils.to_categorical(Y_train)).astype(int)

# Plot data point
d = X_train.shape[1]
dsqrt = int(np.sqrt(d))
plot_test = X_train[4].reshape((dsqrt, dsqrt))
plt.imshow(plot_test)

# Create instance of Sequential class
model = Sequential()

# Add the first (and last) layer for linear classification
model.add(Dense(units=3, input_shape=(d, )))
model.add(Activation("softmax"))

# Optimizer (parameters from the Introduction to Keras slides)
# sgd = SGD(lr=0.01,
#           decay=1e-6,
#           momentum=0.9,
#           nesterov=True)
sgd = SGD()
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])


# Fit the model
test = model.fit(X_train, Y_train_onehot, epochs=1000, batch_size=32)


X_test = mp1.generate_a_disk()
X_test = X_test.reshape(1, X_test.shape[0])
model.predict_proba(X_test)


# Plot the weights
weights = model.get_weights()
w = weights[0]
plt.imshow(w[:, 0].reshape((dsqrt, dsqrt)))
plt.imshow(w[:, 1].reshape((dsqrt, dsqrt)))
plt.imshow(w[:, 2].reshape((dsqrt, dsqrt)))



# COmparison with logistics
logistic = lm.LogisticRegression()

logistic.fit(X_train, Y_train)

logistic.predict_proba(X_test)

coefs = logistic.coef_
plt.imshow(coefs[0, :].reshape((dsqrt, dsqrt)))
plt.imshow(coefs[1, :].reshape((dsqrt, dsqrt)))
plt.imshow(coefs[2, :].reshape((dsqrt, dsqrt)))