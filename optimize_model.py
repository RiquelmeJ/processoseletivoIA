import tensorflow as tf
import os

def main():
    model_path = "model.h5"
    tflite_model_path = "model.tflite"

    if not os.path.exists(model_path):
        print(f"Erro: sem arquivo {model_path}.")
        return

  
    model = tf.keras.models.load_model(model_path)


    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"Modelo otimizado salvo em {tflite_model_path}")

    h5_size = os.path.getsize(model_path) / 1024
    tflite_size = os.path.getsize(tflite_model_path) / 1024
    print(f"Tamanho original: {h5_size:.2f} KB")
    print(f"Tamanho otimizado: {tflite_size:.2f} KB")

if __name__ == "__main__":
    main()
