import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def predict_image(image):
    if image is None:
        return {}

    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))

    image_array = np.array(image).astype("float32") / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    predictions = model.predict(image_array, verbose=0)[0]

    return {
        class_names[i]: float(predictions[i])
        for i in range(len(class_names))
    }

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Gambar Fashion"),
    outputs=gr.Label(num_top_classes=10, label="Hasil Prediksi"),
    title="Fashion-MNIST Image Classification",
    description="Klasifikasi gambar pakaian menggunakan model CNN."
)

if __name__ == "__main__":
    demo.launch()
