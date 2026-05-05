import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)

    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


    model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1)


    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nAcuracia: {test_acc:.4f}")

    model_path = "model.h5"
    model.save(model_path)
    print(f"Modelo salvo: {model_path}")

if __name__ == "__main__":
    main()
