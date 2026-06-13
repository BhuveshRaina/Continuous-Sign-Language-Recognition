import os
import cv2
import numpy as np

def draw_landmarks(image, landmarks, color=(0, 255, 0)):
    """Draws landmarks (dots) on an image."""
    for i in range(0, len(landmarks), 3):
        x = int(landmarks[i] * image.shape[1])
        y = int(landmarks[i+1] * image.shape[0])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 2, color, -1)

def visualize_npy(npy_path, output_base):
    """Visualizes a single .npy file and saves representative frames."""
    data = np.load(npy_path)
    file_name = os.path.splitext(os.path.basename(npy_path))[0]
    # Clean up file name for folder
    folder_name = "".join([c if c.isalnum() or c in " _-" else "_" for c in file_name])
    video_dir = os.path.join(output_base, folder_name)
    os.makedirs(video_dir, exist_ok=True)
    
    num_frames = data.shape[0]
    # Pick 5 representative frames
    indices = np.linspace(0, num_frames - 1, min(num_frames, 5), dtype=int)
    
    for idx in indices:
        frame_data = data[idx]
        # Pose: 0-99, LH: 99-162, RH: 162-225
        pose = frame_data[0:99]
        lh = frame_data[99:162]
        rh = frame_data[162:225]
        
        # Create black image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        draw_landmarks(img, pose, (0, 255, 0))    # Green for pose
        draw_landmarks(img, lh, (0, 0, 255))      # Red for left hand
        draw_landmarks(img, rh, (255, 0, 0))      # Blue for right hand
        
        save_path = os.path.join(video_dir, f"frame_{idx:03d}.png")
        cv2.imwrite(save_path, img)
    
    print(f"Saved {len(indices)} frames for '{file_name}' in: {video_dir}")

if __name__ == "__main__":
    # Sample files from your dataset
    samples = [
        "data/processed_keypoints/he she is my friend/2022IMT-023_v3.npy",
        "data/processed_keypoints/he she is my friend/2022IMT-022_He she is my friend 3.npy",
        "data/processed_keypoints/he she is my friend/2022IMT-022_He she is my friend 2.npy",
        "data/processed_keypoints/he she is my friend/2022IMT-023_v2.npy",
        "data/processed_keypoints/he she is my friend/2022IMT-022_He she is my friend 1.npy"
    ]
    
    output_base = "data/processed_keypoints/visualizations"
    os.makedirs(output_base, exist_ok=True)
    
    print(f"Starting visualization of {len(samples)} files...")
    for sample in samples:
        if os.path.exists(sample):
            visualize_npy(sample, output_base)
        else:
            print(f"File not found: {sample}")
    
    print(f"\nAll visualizations are stored in: {output_base}")
