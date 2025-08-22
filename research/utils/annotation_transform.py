import numpy as np
import os
import cv2
import pandas as pd


# Pick out all key frames of each video according to our pose-level annotation.
def _annotation_transform(root_dir):
    print(f"[DEBUG] Starting annotation transform with root_dir: {root_dir}")
    
    train_type = 'train'
    annotation_name = 'pose_train.csv'
    video_dir = os.path.join(root_dir, 'video', train_type)
    label_filename = os.path.join(root_dir, 'annotation', annotation_name)
    train_save_dir = os.path.join(root_dir, 'extracted')
    save_dir = os.path.join(train_save_dir, train_type)
    
    print(f"[DEBUG] Video directory: {video_dir}")
    print(f"[DEBUG] Label filename: {label_filename}")
    print(f"[DEBUG] Save directory: {save_dir}")
    
    if not os.path.isdir(save_dir):
        print(f"[DEBUG] Creating save directory: {save_dir}")
        os.makedirs(save_dir)
    else:
        print(f"[DEBUG] Save directory already exists: {save_dir}")
    
    print(f"[DEBUG] Reading CSV file: {label_filename}")
    df = pd.read_csv(label_filename)
    print(f"[DEBUG] CSV loaded successfully. Shape: {df.shape}")

    file2label = {}
    num_idx = 3
    print(f"[DEBUG] Processing {len(df)} rows from CSV...")
    
    for i in range(0, len(df)):
        filename = df.loc[i, 'name']
        action_type = df.loc[i, 'type']
        label_tmp = df.values[i][num_idx:].astype(np.float64)
        label_tmp = label_tmp[~np.isnan(label_tmp)].astype(np.int32)
        
        if len(label_tmp) % 2 != 0:
            print(f"[ERROR] Row {i}: filename={filename}, label_tmp length={len(label_tmp)} (odd number)")
            print(f"[ERROR] label_tmp values: {label_tmp}")
            print(f"[ERROR] Raw values: {df.values[i][num_idx:]}")
            raise AssertionError(f"Row {i} has odd number of labels: {len(label_tmp)}")
        
        s1_tmp = label_tmp[::2]
        s2_tmp = label_tmp[1::2]
        file2label[filename] = [s1_tmp, s2_tmp, action_type]
        
        if i < 3:  # Print first 3 entries for debugging
            print(f"[DEBUG] Row {i}: filename={filename}, action_type={action_type}, s1_frames={len(s1_tmp)}, s2_frames={len(s2_tmp)}")
    
    print(f"[DEBUG] Created file2label dictionary with {len(file2label)} entries")

    video_count = 0
    total_videos = len(file2label)
    print(f"[DEBUG] Starting to process {total_videos} videos...")
    
    for video_name in file2label:
        video_count += 1
        video_path = os.path.join(video_dir, video_name)
        print(f'[DEBUG] Processing video {video_count}/{total_videos}: {video_name}')
        print('video_path:', video_path)
        
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file not found: {video_path}")
            continue
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if cap.isOpened():
            frame_count = 0
            while True:
                success, frame = cap.read()
                if success is False:
                    break
                frames.append(frame)
                frame_count += 1
            print(f"[DEBUG] Loaded {frame_count} frames from video")
        else:
            print(f"[ERROR] Failed to open video: {video_path}")
            continue
            
        cap.release()
        s1_label, s2_label, action_type = file2label[video_name]
        print(f"[DEBUG] Action type: {action_type}, S1 labels: {len(s1_label)}, S2 labels: {len(s2_label)}")
        count = 0
        s1_saved = 0
        s2_saved = 0
        
        print(f"[DEBUG] Extracting S1 (salient1) frames...")
        for frame_index in s1_label:
            if frame_index >= len(frames):
                print(f"[WARNING] Frame index {frame_index} out of range (max: {len(frames)-1})")
                continue
            frame_ = frames[frame_index]
            sub_s1_save_dir = os.path.join(save_dir, action_type, 'salient1', video_name)
            if not os.path.isdir(sub_s1_save_dir):
                os.makedirs(sub_s1_save_dir)
            save_path = os.path.join(sub_s1_save_dir, str(count) + '.jpg')
            cv2.imwrite(save_path, frame_)
            count += 1
            s1_saved += 1
        
        print(f"[DEBUG] Saved {s1_saved} S1 frames")
        print(f"[DEBUG] Extracting S2 (salient2) frames...")
        
        for frame_index in s2_label:
            if frame_index >= len(frames):
                print(f"[WARNING] Frame index {frame_index} out of range (max: {len(frames)-1})")
                continue
            frame_ = frames[frame_index]
            sub_s2_save_dir = os.path.join(save_dir, action_type, 'salient2', video_name)
            if not os.path.isdir(sub_s2_save_dir):
                os.makedirs(sub_s2_save_dir)
            save_path = os.path.join(sub_s2_save_dir, str(count) + '.jpg')
            cv2.imwrite(save_path, frame_)
            count += 1
            s2_saved += 1
            
        print(f"[DEBUG] Saved {s2_saved} S2 frames")
        print(f"[DEBUG] Total frames saved for {video_name}: {s1_saved + s2_saved}")
        print("-" * 50)
    
    print(f"[DEBUG] Annotation transform completed! Processed {total_videos} videos.")
