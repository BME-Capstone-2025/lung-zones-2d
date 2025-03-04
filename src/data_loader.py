import tensorflow as tf
import os

def parse_csv_line(line):
    """
    Parses a single CSV line into a filename and keypoints.
    Assumes each CSV has a header:
      filename, x0, x1, ..., x7, y0, y1, ..., y7
    """
    # The first column is a string (filename); the next 16 are floats.
    record_defaults = [[""]] + [[0.0] for _ in range(16)]
    parsed = tf.io.decode_csv(line, record_defaults=record_defaults)
    
    filename = parsed[0]
    # Extract x and y coordinates separately
    x_coords = parsed[1:9]
    y_coords = parsed[9:17]
    # Combine into shape (8, 2): each row is one keypoint (x, y)
    keypoints = tf.stack([x_coords, y_coords], axis=1)
    return filename, keypoints

def load_image_and_keypoints(frames_dir, video_name, filename, keypoints, image_size=(224, 224)):
    """
    Constructs the full image path from the base frames directory, video name, and filename,
    loads the image, and resizes it.
    """
    video_path = os.path.join(frames_dir, video_name)
    video_path_tf = tf.constant(video_path)
    image_path = tf.strings.join([video_path_tf, filename], separator=os.sep)
    
    # Read, decode, and resize the image.
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    return image, keypoints

def create_dataset_from_csv(csv_file, frames_dir, video_name, image_size=(224, 224), augment_fn=None):
    """
    Creates a tf.data.Dataset from one CSV file.
    Optionally applies an augmentation function to the image and keypoints.
    """
    dataset = tf.data.TextLineDataset(csv_file)
    dataset = dataset.skip(1)
    dataset = dataset.map(parse_csv_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        lambda filename, keypoints: load_image_and_keypoints(frames_dir, video_name, filename, keypoints, image_size),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # If an augmentation function is provided, apply it.
    if augment_fn is not None:
        dataset = dataset.map(
            lambda image, keypoints: augment_fn(image, keypoints, image_size),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    return dataset

def get_dataset(base_dir, batch_size=32, image_size=(224, 224), shuffle=True, buffer_size=1000, augment_fn=None):
    """
    Constructs a dataset from your custom directory structure.
    
    Expects the following structure:
      base_dir/
        └── label/
            ├── frame_labels/    # Contains CSV files (one per video)
            └── frames/          # Contains subdirectories (one per video) with images
              
    Each CSV file name (e.g. video1.csv) is used to locate images in:
        label/frames/video1/
    """
    csv_dir = os.path.join(base_dir, "label", "frame_labels")
    frames_dir = os.path.join(base_dir, "label", "frames")
    
    # List all CSV files in the frame_labels directory
    csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith(".csv")]
    
    # Create a dataset for each CSV file and collect them in a list
    datasets = []
    for csv_file in csv_files:
        # Use the CSV file name (without extension) as the video name
        video_name = os.path.splitext(os.path.basename(csv_file))[0]
        ds = create_dataset_from_csv(csv_file, frames_dir, video_name, image_size, augment_fn)
        datasets.append(ds)
    
    # Concatenate all individual datasets into one
    dataset = datasets[0]
    for ds in datasets[1:]:
        dataset = dataset.concatenate(ds)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset



def random_brightness(image, keypoints, image_size, max_delta=0.2):
    # Apply random brightness with 50% chance
    apply = tf.random.uniform([]) < 0.5
    image = tf.cond(apply, lambda: tf.image.random_brightness(image, max_delta), lambda: image)
    return image, keypoints

def random_rotation(image, keypoints, image_size, max_angle=10):
    # Rotate the image by a random angle (in degrees) with 50% chance
    apply = tf.random.uniform([]) < 0.5
    angle = tf.random.uniform([], minval=-max_angle, maxval=max_angle) * (3.14159265 / 180.0)
    
    def rotate():
        rotated_image = tfa.image.rotate(image, angle)  # Requires tensorflow-addons
        # Note: You must also rotate the keypoints appropriately.
        # This is a placeholder: you'll need to implement the coordinate transformation.
        rotated_keypoints = keypoints  # Replace with proper rotation of keypoints.
        return rotated_image, rotated_keypoints
    
    image, keypoints = tf.cond(apply, rotate, lambda: (image, keypoints))
    return image, keypoints

def apply_chain_augmentations(image, keypoints, image_size):
    # Chain multiple augmentation functions
    image, keypoints = random_flip_horizontal(image, keypoints, image_size)
    image, keypoints = random_brightness(image, keypoints, image_size)
    # Uncomment the next line if you have tensorflow-addons and implement keypoint rotation.
    # image, keypoints = random_rotation(image, keypoints, image_size)
    return image, keypoints

# ---------------------------------------------------------------------------
# Example usage:
if __name__ == '__main__':
    base_directory = 'C:\Users\micha\Desktop\BME461'  # Update this path accordingly.
    # Pass your augmentation function here (or use None if no augmentations are desired)
    ds = get_dataset(base_directory, batch_size=16, image_size=(224, 224), augment_fn=random_brightness)
    
    # Iterate over one batch to test the pipeline
    for images, keypoints in ds.take(1):
        print("Batch of images shape:", images.shape)
        print("Batch of keypoints shape:", keypoints.shape)
