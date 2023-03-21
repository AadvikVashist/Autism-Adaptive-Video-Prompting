import noduro
import time
import cv2
import numpy as np
import os
import read_settings as tools

ADJUSTER = 1.1
OPTIMAL_FPS = 20
from modeling.gesture.gesture_tracker import gesture_tracker
from modeling.gesture.realtime_gesture_analysis import realtime_gesture_analysis
class frame_skip_reccomendation(gesture_tracker):
    def video_analysis(self, video, skip_range : list = [1,11]):
        def video_dimensions_fps(videofile):
            vid = cv2.VideoCapture(videofile) #vid capture object
            _,frame = vid.read()
            height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            fps = vid.get(cv2.CAP_PROP_FPS)
            vid.release()
            return int(width),int(height),fps
        ret = []
        for skip in range(*skip_range):
            self.capture = cv2.VideoCapture(video)
            self.vid_info = video_dimensions_fps(video)
            self.index = 0
            self.re = []
            self.processed_frame = {}
            self.visibilty_dict = {}
            self.etc["frame_skip"] = skip
            self.looping_analysis(videoCapture = self.capture, video_shape = None, fps = None, result_vid = None, starting_vid = None, frame_skip = skip, save_pose = False, standardize_pose = True,save_frames = False)  
            self.re = [self.re[r] - self.re[r-1] for r in range(1,len(self.re))][1::]
            ret.append(self.re)
        return ret, np.swapaxes([(self.index+1, 1/np.mean(av)) for self.index, av in enumerate(ret)],1,0)
    def while_processing(self, frame):
        a = super().while_processing(frame)
        self.re.append(time.time)
        self.index += 1
        return a
class frame_skip_reccomendation_with_classification(realtime_gesture_analysis):
    def video_analysis(self, video, skip_range : list = [1,11]):
        def video_dimensions_fps(videofile):
            vid = cv2.VideoCapture(videofile) #vid capture object
            _,frame = vid.read()
            height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            fps = vid.get(cv2.CAP_PROP_FPS)
            vid.release()
            return int(width),int(height),fps
        ret = []
        for skip in range(*skip_range):
            self.capture = cv2.VideoCapture(video)
            self.vid_info = video_dimensions_fps(video)
            self.index = 0
            self.re = []
            self.processed_frame = {}
            self.visibilty_dict = {}
            self.etc["frame_skip"] = skip
            self.looping_analysis(videoCapture = self.capture, video_shape = None, fps = None, result_vid = None, starting_vid = None, frame_skip = skip, save_pose = False, standardize_pose = True,save_frames = False)  
            self.re = [self.re[r] - self.re[r-1] for r in range(1,len(self.re))][1::]
            ret.append(self.re)
        return ret, np.swapaxes([(self.index+1, 1/np.mean(av)) for self.index, av in enumerate(ret)],1,0)
    def while_processing(self, frame):
        a = super().while_processing(frame)
        self.re.append(time.time)
        self.index += 1
        return a
file = noduro.subdir_path("data/raw/fps_tester/test1.mp4")

def get_suggested_frame_skip(with_realtime = False):
    if with_realtime:
        a = frame_skip_reccomendation(frameskip= True)
    else: 
        a = frame_skip_reccomendation_with_classification
    timeskip, averages = a.video_analysis(file, )
    times = str(int(time.time()))

    average_csv_file = noduro.subdir_path("data/analyzed/frame_skipping", "averages.csv")        
    noduro.write_csv(average_csv_file, averages, True, False)
    print("saved averages to", average_csv_file)

    raw_csv_file = noduro.subdir_path("data/analyzed/frame_skipping", "times"+ times + ".csv")        
    if os.path.exists(raw_csv_file) == False:
        noduro.write_csv(raw_csv_file, [1,2,3,4,5,6,7,8,9,10], True,False)
        print("made", raw_csv_file, "file")
    timeskip = np.swapaxes(timeskip,1,0)
    noduro.write_csv(raw_csv_file, timeskip, False, False)
    print("saved timeskip to", raw_csv_file)
    return averages
def suggestion(average_list):
    list_adjusted = np.asarray(average_list[1])/ADJUSTER
    reccomendation = np.argmin([abs(l - OPTIMAL_FPS) for l in list_adjusted])
    tools.set_points(average_list[0][reccomendation])
    print("reccomendation is one analysis every",average_list[0][reccomendation], "frames which is",average_list[1][reccomendation] ,"fps")
if __name__ == '__main__':
    averages = get_suggested_frame_skip(with_realtime = True)
    suggestion(averages)