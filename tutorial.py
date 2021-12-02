import tensorflow as tf
import numpy as np

dataset = np.loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
X = dataset[:, 0:8]
y = dataset[:, 8]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(12, input_dim=8, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=150, batch_size=10, validation_split=1/3)

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))