import pandas as pd
import numpy as np
import os
import cv2
try:
    from mediapipe.python.solutions import pose as mp_pose
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("[WARNING] MediaPipe not available. Some functions may not work.")
    MEDIAPIPE_AVAILABLE = False
    mp_pose = None
import torch.onnx
import time
import yaml
import argparse
from utils import _annotation_transform, _generate_for_train, _generate_csv_label

torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    old_time = time.time()

    if os.path.isfile(args.config):
        with open(args.config, "r") as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)
    else:
        raise ValueError("Config file does not exist.")

    root_dir = config['dataset']['dataset_root_dir']
    csv_label_path = config['dataset']['csv_label_path']

    print('start annotation transform')
    print(f"[DEBUG] Root directory: {root_dir}")
    print(f"[DEBUG] CSV label path: {csv_label_path}")
    _annotation_transform(root_dir)
    print('[INFO] Annotation transform completed successfully!')

    print('start generate csv label')
    _generate_csv_label(root_dir, csv_label_path)
    print('[INFO] CSV label generation completed successfully!')

    print('start generate for train')
    _generate_for_train(root_dir)
    print('[INFO] Training data generation completed successfully!')

    current_time = time.time()
    print('time: ' + str(current_time - old_time) + 's')
    print('[INFO] All preprocessing steps completed successfully!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate our PoseRAC')
    parser.add_argument('--config', type=str, metavar='DIR',
                        help='path to a config file')
    args = parser.parse_args()
    main(args)
