import os
import urllib.request

# Define model URL and target path
MODEL_URL = "https://github.com/hiennguyen9874/yolov7-face-detection/releases/download/v0.2/yolov7-tiny76.pt"
WEIGHTS_DIR = "weights"
MODEL_PATH = os.path.join(WEIGHTS_DIR, "yolov7-tiny.pt")

def download_model():
    # Create weights directory if it does not exist
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        print(f"Model file '{MODEL_PATH}' already exists.")
        return
    try:
        print(f"Downloading model file to '{MODEL_PATH}'...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Download complete: '{MODEL_PATH}'")
    except Exception as e:
        print(f"Error while downloading the model: {e}")

if __name__ == "__main__":
    download_model()
