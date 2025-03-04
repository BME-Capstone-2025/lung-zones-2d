import argparse
import csv
import cv2
import numpy as np
import tensorflow as tf

# Define keypoint labels
LABEL_NAMES = ["RACW", "RAAXL", "RCosto", "RPLAPs", "LACW", "LAAXL", "LCosto", "LPlaps"]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay predicted keypoints (red) and ground truth keypoints (green) on a video"
    )
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
    parser.add_argument("--gt_csv", type=str, default=None,
                        help="Path to a CSV file containing ground truth keypoints")
    return parser.parse_args()

def overlay_keypoints(frame, keypoints, color, radius=5, thickness=-1):
    """
    Draw circles and labels on the frame at the given keypoint locations.
    
    Args:
        frame: Image (numpy array in BGR).
        keypoints: Array of shape (num_keypoints, 2) with (x, y) coordinates.
        color: Color tuple in BGR (e.g., (0,0,255) for red).
        radius: Circle radius.
        thickness: Circle thickness (-1 for filled).
    """
    for i, point in enumerate(keypoints):
        x, y = int(point[0]), int(point[1])
        if x <= -0.9 and y <= -0.9:
            continue
        cv2.circle(frame, (x, y), radius, color, thickness)
        cv2.putText(frame, LABEL_NAMES[i], (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def load_ground_truth(gt_csv_path, num_keypoints):
    """
    Loads ground truth keypoints from a CSV file.
    
    Assumes each CSV row is:
      filename, x0, x1, ..., x7, y0, y1, ..., y7
    Returns:
      A list of numpy arrays of shape (num_keypoints, 2)
      (assumed to be normalized coordinates).
    """
    gt_list = []
    with open(gt_csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header
        for row in reader:
            if len(row) < 1 + num_keypoints * 2:
                continue
            try:
                vals = np.array(list(map(float, row[1:])))
            except ValueError:
                continue
            gt = vals.reshape(2, num_keypoints).T
            gt_list.append(gt)
    return gt_list

def main():
    args = parse_args()
    
    # Get the model input dimensions from arguments.
    input_height, input_width = (int(x) for x in args.input_size.split(","))
    model_input_size = (input_height, input_width)
    
    # Load your trained model.
    model = tf.keras.models.load_model(args.model_path)
    
    # Open the input video.
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print("Error: Could not open input video.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (orig_width, orig_height))
    
    # Load ground truth keypoints if provided.
    gt_keypoints_list = None
    if args.gt_csv:
        gt_keypoints_list = load_ground_truth(args.gt_csv, args.num_keypoints)
        print(f"Loaded ground truth for {len(gt_keypoints_list)} frames from {args.gt_csv}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save a copy of the original frame (for overlay) in BGR.
        frame_overlay = frame.copy()
        
        # --- Mimic the data loader pipeline ---
        # Encode the frame to JPEG in memory.
        ret2, jpeg_buffer = cv2.imencode('.jpg', frame)
        if not ret2:
            print("Error encoding frame to JPEG")
            continue
        image_bytes = jpeg_buffer.tobytes()
        # Decode the JPEG using tf.image.decode_jpeg (this is what your data loader does)
        frame_tensor = tf.image.decode_jpeg(image_bytes, channels=3)
        # Resize using tf.image.resize (same as in your loader)
        frame_resized = tf.image.resize(frame_tensor, model_input_size,
                                        method=tf.image.ResizeMethod.BILINEAR)
        # Expand dimensions to create batch dimension.
        input_tensor = tf.expand_dims(frame_resized, axis=0)
        # If your training normalized images (e.g., dividing by 255), do it here:
        # input_tensor = input_tensor / 255.0
        
        # --- Run inference ---
        predictions = model.predict(input_tensor)
        # Reshape predictions to (num_keypoints, 2). Assumes model outputs 16 values.
        pred_keypoints = predictions.reshape(2, args.num_keypoints).T
        
        # Scale keypoints from normalized coordinates to original frame dimensions.
        pred_keypoints[:, 0] *= orig_width
        pred_keypoints[:, 1] *= orig_height
        
        # Overlay predicted keypoints in red.
        frame_overlay = overlay_keypoints(frame_overlay, pred_keypoints, color=(0, 0, 255))
        
        # Overlay ground truth (if provided) in green.
        if gt_keypoints_list is not None and frame_count < len(gt_keypoints_list):
            gt_keypoints = np.array(gt_keypoints_list[frame_count])
            gt_keypoints[:, 0] *= orig_width
            gt_keypoints[:, 1] *= orig_height
            frame_overlay = overlay_keypoints(frame_overlay, gt_keypoints, color=(0, 255, 0))
            # Debug: print keypoints info.
            print(f"Frame {frame_count}:")
            print("Ground truth:", gt_keypoints_list[frame_count])
            print("Predictions:", predictions)
        
        out.write(frame_overlay)
        
        if args.display:
            cv2.imshow("Overlay", frame_overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
