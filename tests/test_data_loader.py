import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import get_dataset, random_brightness


if __name__ == '__main__':
    base_directory = r'C:\Users\micha\Desktop\BME461'  # Update this path accordingly.
    # Pass your augmentation function here (or use None if no augmentations are desired)
    ds = get_dataset(base_directory, batch_size=16, image_size=(224, 224), augment_fn=random_brightness)
    
    # Iterate over one batch to test the pipeline
    for images, keypoints in ds.take(1):
        print("Batch of images shape:", images.shape)
        print("Batch of keypoints shape:", keypoints.shape)