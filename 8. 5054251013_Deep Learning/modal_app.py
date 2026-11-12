import modal

app = modal.App("fashion-mnist-classifier")

# Menyiapkan lingkungan Python di cloud Modal
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "tensorflow==2.21.0",
        "gradio==6.31.0",
        "numpy",
        "pillow",
        "fastapi[standard]"
    )
    .add_local_file(
        "deployment/model.keras",
        remote_path="/root/model.keras"
    )
)

# Import library untuk lingkungan cloud
with image.imports():
    import gradio as gr
    import tensorflow as tf
    import numpy as np
    from PIL import Image, ImageOps
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app


@app.function(image=image, max_containers=1)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def ui():

    # Memuat model CNN
    model = tf.keras.models.load_model("/root/model.keras")

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

    return mount_gradio_app(
        app=FastAPI(),
        blocks=demo,
        path="/"
    )
