#!/usr/bin/env python

from tensorflow.python import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# normalize for better accuracy
x_train = x_train / 255
x_test = x_test /255


early_stop = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation="softmax")
    
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=10, callbacks=[early_stop])

model.evaluate(x_test,y_test)

model.save("handwritten_digi_classifier.keras")

