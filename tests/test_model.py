import sys
import os
import tensorflow as tf
from os.path import join

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import get_dataset
from model import get_model


CHECKPOINT_DIR = "./models/checkpoints"

if __name__=="__main__": 
    base_directory = r'C:\Users\micha\Desktop\BME461'  # Update this path accordingly.
    ds = get_dataset(base_directory)
    
    # Ensure loading the latest checkpoint
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if os.path.isfile(join(CHECKPOINT_DIR, f))]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")

    last_checkpoint_file = sorted(
        checkpoint_files, key=lambda x: os.path.getmtime(join(CHECKPOINT_DIR, x))
    )[-1]

    model = tf.keras.models.load_model(join(CHECKPOINT_DIR, last_checkpoint_file))

    print(f"Restored from checkpoint: {last_checkpoint_file}")

    for images, keypoints in ds.take(1):
        loss, metrics = model.evaluate(images, keypoints)
        print(f"Loss: {loss}, Metrics: {metrics}")
