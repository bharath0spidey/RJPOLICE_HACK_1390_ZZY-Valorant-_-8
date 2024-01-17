import os
from collections import OrderedDict
import whisper
import torch

from src.commons import WHISPER_MODEL_WEIGHTS_PATH

def download_whisper():
    try:
        model = whisper.load_model("tiny.en")
        return model
    except Exception as e:
        print(f"Error downloading the Whisper model: {e}")
        return None

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_and_save_encoder(model):
    if model is not None:
        model_ckpt = OrderedDict()
        model_ckpt['model_state_dict'] = OrderedDict()

        for key, value in model.encoder.state_dict().items():
            model_ckpt['model_state_dict'][f'encoder.{key}'] = value

        model_ckpt['dims'] = model.dims

        # Ensure the directory exists before saving
        ensure_directory_exists(os.path.dirname(WHISPER_MODEL_WEIGHTS_PATH))

        torch.save(model_ckpt, WHISPER_MODEL_WEIGHTS_PATH)
        print(f"Saved encoder at '{WHISPER_MODEL_WEIGHTS_PATH}'")
    else:
        print("Cannot extract and save encoder. Model is None.")

if __name__ == "__main__":
    model = download_whisper()

    if model is not None:
        print("Downloaded Whisper model!")
        extract_and_save_encoder(model)
    else:
        print("Exiting script due to download error.")
