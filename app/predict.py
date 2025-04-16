import numpy as np
from loader import load_model_from_path
from PIL import Image, ImageOps

model_path = 'models/handwritten_digi_classifier.keras'

model = load_model_from_path(model_path)


def predict(image) -> int:
    image = Image.open(image)
    image = image.convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert the image
    image = image.resize((28, 28))

    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize the image
    img_array = img_array.reshape(1, 28, 28)  # Reshape for the model

    if model is None:
        raise Exception("Model could not be loaded. Please check the model path and format.")

    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction[0])
    return predicted_class.astype(int)


