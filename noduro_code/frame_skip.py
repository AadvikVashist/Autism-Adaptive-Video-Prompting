from modeling.gesture.gesture_tracker import gesture_tracker
import noduro
import time
import cv2
import numpy as np
import os
import read_settings as tools

ADJUSTER = 1.1
OPTIMAL_FPS = 20

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
            index = 0
            re = []
            self.processed_frame = {}
            self.visibilty_dict = {}
            self.etc["frame_skip"] = skip
            start = time.time()
            while True:
                _, frame = self.capture.read()
                if frame is None:
                    cv2.destroyAllWindows()
                    break
                if index % skip == 0: 
                    frame = self.per_frame_analysis(frame, True)
                re.append(time.time())
                index += 1
                cv2.imshow("frames",frame) #show frame
                cv2.waitKey(1)
            re = [re[r] - re[r-1] for r in range(1,len(re))][1::]
            ret.append(re)
        return ret, np.swapaxes([(index+1, 1/np.mean(av)) for index, av in enumerate(ret)],1,0)

file = noduro.subdir_path("data/raw/fps_tester/test1.mp4")

def get_suggested_frame_skip():
    a = frame_skip_reccomendation(frameskip= True)
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
    averages = get_suggested_frame_skip()
    suggestion(averages)