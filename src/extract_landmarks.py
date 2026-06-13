import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

# MediaPipe
mp_holistic = mp.solutions.holistic

# =========================
# CONFIG
# =========================
PROJECT_ROOT = "/content/drive/MyDrive/NLP_CSL_Recognition"
CSV_PATH = os.path.join(PROJECT_ROOT, "metadata", "video_metadata.csv")
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "processed_keypoints")

GROUP_COLUMN = "group"   # change this if your CSV column has another name
START_GROUP = 1      # Group_10
END_GROUP = 32   # Group_20

TARGET_FPS = 15
# =========================


def extract_keypoints(results):
    pose = (
        np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(33 * 3)
    )
    lh = (
        np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(21 * 3)
    )
    rh = (
        np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(21 * 3)
    )
    return np.concatenate([pose, lh, rh])


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return results


def process_single_video(actual_video_path, save_path):
    if os.path.exists(save_path):
        return "skipped"

    cap = cv2.VideoCapture(actual_video_path)
    if not cap.isOpened():
        return "failed"

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = TARGET_FPS

        skip_rate = max(1, int(fps / TARGET_FPS))

        keypoints_list = []
        frame_idx = 0

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % skip_rate == 0:
                    results = mediapipe_detection(frame, holistic)
                    keypoints_list.append(extract_keypoints(results))

                frame_idx += 1

        cap.release()

        if keypoints_list:
            np.save(save_path, np.array(keypoints_list))
            return "done"

        return "failed"

    except Exception:
        cap.release()
        return "failed"


def run_processing(group_column, start_group, end_group):
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    if group_column not in df.columns:
        raise ValueError(f"Column '{group_column}' not found in CSV")

    if "video_path" not in df.columns:
        raise ValueError("CSV must contain a 'video_path' column")

    if "sentence_label" not in df.columns:
        raise ValueError("CSV must contain a 'sentence_label' column")

    if "student_id" not in df.columns:
        raise ValueError("CSV must contain a 'student_id' column")

    # Extract numeric part from values like Group_13
    group_num = pd.to_numeric(
        df[group_column].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce"
    )

    df = df.copy()
    df["_group_num"] = group_num
    df = df.dropna(subset=["_group_num"]).copy()
    df["_group_num"] = df["_group_num"].astype(int)

    df_filtered = df[
        (df["_group_num"] >= start_group) &
        (df["_group_num"] <= end_group)
    ].copy()

    if df_filtered.empty:
        print(f"No videos found for Group_{start_group} to Group_{end_group}")
        return

    # Sort by group so messages are group-wise
    df_filtered = df_filtered.sort_values(["_group_num", "student_id"]).reset_index(drop=True)

    print(f"Groups selected: Group_{start_group} to Group_{end_group}")
    print(f"Total videos: {len(df_filtered)}\n")

    results_summary = {"done": 0, "skipped": 0, "failed": 0}

    with tqdm(total=len(df_filtered), desc="Extracting Videos", unit="video", dynamic_ncols=True) as pbar:
        for group_value, group_df in df_filtered.groupby("_group_num", sort=True):
            group_name = f"Group_{group_value}"
            tqdm.write(f"Processing {group_name} ({len(group_df)} videos)")

            for _, row in group_df.iterrows():
                rel_path = str(row["video_path"]).split("NLP_CSL_Recognition/")[-1]
                actual_path = os.path.join(PROJECT_ROOT, rel_path)

                save_dir = os.path.join(OUTPUT_BASE_DIR, str(row["sentence_label"]))
                os.makedirs(save_dir, exist_ok=True)

                video_base = os.path.splitext(os.path.basename(actual_path))[0]
                save_name = f"{row['student_id']}_{video_base}.npy"
                save_path = os.path.join(save_dir, save_name)

                status = process_single_video(actual_path, save_path)
                if status in results_summary:
                    results_summary[status] += 1

                pbar.update(1)

            tqdm.write(f"Finished {group_name}\n")

    print("===== SUMMARY =====")
    print(f"Processed: {results_summary['done']}")
    print(f"Skipped  : {results_summary['skipped']}")
    print(f"Failed   : {results_summary['failed']}")
    print("===================")


if __name__ == "__main__":
    run_processing(GROUP_COLUMN, START_GROUP, END_GROUP)