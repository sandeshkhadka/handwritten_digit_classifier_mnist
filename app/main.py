import sys

from predict import predict

image_path = sys.argv[1]

if __name__ == "__main__":
    try:
        predicted_class = predict(image_path)
        print(f"Predicted class: {predicted_class}")
    except Exception as e:
        print(f"Error during prediction: {e}")
