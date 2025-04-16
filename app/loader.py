from tensorflow.keras.models import load_model


def load_model_from_path(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None
