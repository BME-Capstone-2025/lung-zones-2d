import argparse
import cv2
import numpy as np
import tensorflow as tf

LABEL_NAMES = ["RACW", "RAAXL", "RCosto", "RPLAPs", "LACW", "LAAXL", "LCosto", "LPlaps"]


def parse_args():
    parser = argparse.ArgumentParser(description="Overlay predicted keypoints on a video")
    parser.add_argument("--input_video", type=str, required=True,
                        help="Path to the input video file")
    parser.add_argument("--output_video", type=str, required=True,
                        help="Path to the output video file")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model (.h5 file)")
    parser.add_argument("--input_size", type=str, default="224,224",
                        help="Model input size as comma-separated values (height,width)")
    parser.add_argument("--num_keypoints", type=int, default=8,
                        help="Number of keypoints predicted by the model")
    parser.add_argument("--display", action="store_true",
                        help="If set, display frames during processing")
    return parser.parse_args()

def overlay_keypoints(frame, keypoints, color=(0, 0, 255), radius=5, thickness=-1):
    """
    Draw circles on the frame at the given keypoints.
    
    Args:
        frame: The original image (numpy array).
        keypoints: Array of shape (num_keypoints, 2) containing (x, y) coordinates.
        color: Color of the circles (B, G, R).
        radius: Radius of the circles.
        thickness: Thickness of the circles; use -1 for filled circles.
    """
    for i, point in enumerate(keypoints):
        x, y = int(point[0]), int(point[1])
        # Skip if the keypoint is (-1, -1) (not visible)
        if x == -1 and y == -1:
            continue
        cv2.circle(frame, (x, y), radius, color, thickness)
        cv2.putText(frame, LABEL_NAMES[i], (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def main():
    args = parse_args()
    
    # Parse the input_size argument into a tuple (height, width)
    input_height, input_width = (int(x) for x in args.input_size.split(","))
    model_input_size = (input_height, input_width)
    
    # Load the trained model
    model = tf.keras.models.load_model(args.model_path)
    
    # Open the input video file
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print("Error: Could not open input video.")
        return
    
    # Get original video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Adjust codec if needed
    
    # Initialize the video writer
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (orig_width, orig_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame:
        # 1. Resize the frame to the model input size
        resized_frame = cv2.resize(frame, (model_input_size[1], model_input_size[0]))
        # 2. Convert the image to float32 and add batch dimension
        input_tensor = tf.convert_to_tensor(resized_frame, dtype=tf.float32)
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        # If your model was trained with a specific preprocessing (e.g., normalization),
        # apply it here (for example, using tf.keras.applications.mobilenet_v2.preprocess_input).
        
        # Run inference on the frame
        predictions = model.predict(input_tensor)
        # Reshape predictions into (num_keypoints, 2)
        keypoints = predictions.reshape(args.num_keypoints, 2)
        
        # Scale the keypoints to the original frame size
        scale_x = orig_width
        scale_y = orig_height
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y
        
        # Overlay the keypoints on the original frame
        frame_with_keypoints = overlay_keypoints(frame.copy(), keypoints)
        
        # Write the processed frame to the output video
        out.write(frame_with_keypoints)
        
        if args.display:
            cv2.imshow("Keypoints Overlay", frame_with_keypoints)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
