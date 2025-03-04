import sys
import os
import tensorflow as tf
from os.path import join

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import get_dataset
from model import get_model


CHECKPOINT_DIR = "./models/checkpoints/overfitsaad"
LABEL_NAMES = ["RACW", "RAAXL", "RCosto", "RPLAPs", "LACW", "LAAXL", "LCosto", "LPlaps"]

if __name__=="__main__": 
    base_directory = "C:\\Users\\micha\\Desktop\\data_03032025"  # Update this path accordingly.
    ds = get_dataset(base_directory, shuffle=False)
    
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
        pred = model.predict(images)
        for i in range(keypoints.shape[0]): 
            print(f"{'Label':^20}  | {'True':^20} | {'Pred':^20}   Frame: {i}")
            print("-" * 65)
            for d in range(8): 
                print(f"{LABEL_NAMES[d]:^20}    {keypoints[i, d]:^9.3f}, {keypoints[i, d+8]:^9.3f} | {pred[i,d]:^9.3f}, {pred[i,d+8]:^9.3f}")
            input("...")

