#size = 1/distance.
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import noduro
import time
from csv import writer
import pandas as pd
from modeling.gesture.gesture_tracker import gesture_tracker
import noduro_code.read_settings as read_settings
import os
import modeling.gesture.pose_manipulation.pose_standardizer as pose_standardizer
class calibrate_pose(gesture_tracker):
    def start(self):
        self.calibrated_values = []
    def while_processing(self, frame, process):
        if process:
            try:
                self.calibrated_values.append(pose_standardizer.calibrate_pose_using_eye(pose_standardizer.convert_holistic_to_dict(self.processed_frame["holistic"]), self.gesture_point_dict))
            except:
                pass
            return frame
    def end(self):
        der = pose_standardizer.derive_calibration(self.calibrated_values)
        write_results(self.video_file,(der,self.calibrated_values))
        print(der)

def write_results(filename, results : list, results_json = noduro.subdir_path("data/analyzed/pose_standardization/standardizers.json")):
    times = time.time()
    res = {''.join(os.path.splitext(filename)[0].split('/')[-1::]) + "_" +  str(times) + "_averages": results[0], ''.join(os.path.splitext(filename)[0].split('/')[-1::]) + "_" + str(times) + "_raw": results[1]}
    noduro.write_json(results_json,res,False)

def results_to_settings(results_csv = noduro.subdir_path("data/analyzed/pose_standardization/standardizers.json")):
    json_file = noduro.read_json(results_csv,False)
    json_file = [r for key, r in json_file.items() if "raw" in key]
    scale = np.mean(json_file)
    read_settings.set_scale_factor(scale)

if __name__ == "__main__":
    a = calibrate_pose()

    files = noduro.get_dir_files("data/raw/pose_standardizer", True)
    for file in files:
        a.video_analysis(file)
    results_to_settings()