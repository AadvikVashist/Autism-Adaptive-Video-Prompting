#size = 1/distance.
import numpy as np
import matplotlib.pyplot as plt
import noduro
import time
from csv import writer
import noduro_code.read_settings as read_settings
import noduro
from sklearn.impute import SimpleImputer
#import pose_standardizer

try:
    SCALE = read_settings.get_scale_factor()
except:
    print("no scale")

DEFAULT_FILE,_ = read_settings.get_settings()
DEFAULT_FILE = DEFAULT_FILE["filesystem"]["pose_standardization"] #in settings
DICT = noduro.read_json(DEFAULT_FILE, True)
IMPUTER1 = SimpleImputer(missing_values=np.nan,  strategy='median')
IMPUTER2 = SimpleImputer(missing_values=np.nan,  strategy='constant', fill_value = 0.5)
def convert_holistic_to_dict(holistic_values):
    return {"face" : holistic_values.face_landmarks, "pose" : holistic_values.pose_landmarks, "left_hand" : holistic_values.left_hand_landmarks, "right_hand" : holistic_values.right_hand_landmarks}

def iter_landmarks(landmark_list,feature_dict):
        new = []
        for index, (key, value) in enumerate(feature_dict.items()):
            #check if dict landmark_list has the key, fill it with sklearn imputers
            if landmark_list is None:
                value = [(np.nan,np.nan,np.nan) for v in value]
            else:
                value = [(landmark_list[val].x,landmark_list[val].y,landmark_list[val].z) for val in value]
            new.extend(value)
        return new

def filter_body_part(landmarks, ref_dict : dict):
    if landmarks is not None:
        landmarks = landmarks.landmark
    val = iter_landmarks(landmarks,ref_dict)
    return val

def filter_body_parts(landmarks : dict, ref : dict):
    ret = []
    for key, value in ref.items():
        ret.append(filter_body_part(landmarks[key],value))
    return ret

def convert_holistic_to_parts(landmarks,ref_dict):
    return filter_body_parts(convert_holistic_to_dict(landmarks), ref_dict)

def display_pose_direct(points):
    fig = plt.figure()
    # points = convert_holistic_to_parts(points, gesture_point_dict)
    ax = plt.axes(projection='3d')
    for point in points:
        point = np.array(point)
        ax.scatter3D(point[:,0],point[:,2]-1,point[:,1]*-1)
    ax.set_xlim([0,1])
    ax.set_ylim([-1,0])
    ax.set_zlim([-1,0])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

def distance(point1,point2):
    dist = point2-point1
    return np.sqrt(dist[0]**2 + dist[1]**2 + dist[2]**2)

def derive_calibration(values):
    # file = np.array(pd.read_csv(file, header = None))
    ret = [np.min(values), np.mean(values), np.max(values), np.std(values), len(values)]
    return ret
def calibrate_pose_using_eye_and_chest(points): #use the eye for scaling, and the chest as a centering value
    points["face"] = points["face"]
    left = DICT["scalar"]["face"]["left_iris"]; left = [(points["face"].landmark[l].x,points["face"].landmark[l].y,points["face"].landmark[l].z) for l in left]
    right =  DICT["scalar"]["face"]["right_iris"]; right = [(points["face"].landmark[r].x,points["face"].landmark[r].y,points["face"].landmark[r].z) for r in right]
    left = np.mean(left, axis = 0)
    right = np.mean(right, axis = 0)
    dist = distance(left,right)

    chest = DICT["center"]["pose"]["chest"]; center = np.mean([(points["pose"].landmark[c].x,points["pose"].landmark[c].y,points["pose"].landmark[c].z) for c in chest],axis = 0)
    return dist/SCALE,center #return ratio between size of current image and SCALE value. 
def center_and_scale_from_raw(points, gpdict, moving_average = None): #derived point values and the gesture point dictionary
    if "face" in points.keys() and "pose" in points.keys():
        curr_scale,center = calibrate_pose_using_eye_and_chest(points)
    else:
        print("no face and/or pose")
    if moving_average is not None:
        set_scale = np.mean((curr_scale,np.mean(moving_average)))
    else:
        set_scale = curr_scale
    ret = []
    s = time.time()
    for key, value in gpdict.items():
        value = [x for v in value.values() for x in v]
        if points[key] == None:
            value = [(np.nan,np.nan,np.nan) for v in value]
        else:
            value = [((points[key].landmark[val].x,points[key].landmark[val].y,points[key].landmark[val].z)-center)/curr_scale for val in value]
        ret.append(value)
    return ret,curr_scale

def flatten_gesture_point_dict_to_list(gpdict):
    ret = []
    for key, value in gpdict.items():
        value = [x for v in value.values() for x in v]
        ret.append(value)
    return ret


def flatten_gesture_point_dict_keys_to_list(gpdict):
    ret = {}
    for key, value in gpdict.items():
        ret[key] = [x for v in value.values() for x in v]
    return ret
def flatten_3d_to_1d(points : list):
    return np.array([x for v in points for x in v]).flatten()
# def standardize_pose(points):

def fill_nans_with_imputer_for_sklearn_regression(list_3d : list,multiple_frames : bool = True) -> list:
        #check if the class has a variable named self.imputer
    if not multiple_frames:
        list_1d = [[x for v in list_3d for x in flatten_3d_to_1d(v)]]
        return IMPUTER2.fit_transform(list_1d)
    else:
        list_1d = [flatten_3d_to_1d(d) for d in list_3d]
        return IMPUTER1.fit_transform(list_1d)

def convert_processed_frame_to_filtered_parts(processed_frame : dict,gpdict) -> dict:
    gesture_dic = convert_holistic_to_dict(processed_frame["holistic"])
    filter_body_parts(gesture_dic, gpdict)
def convert_processed_frame_to_standardized_parts(processed_frame : dict,gpdict, moving_average) -> dict:
    gesture_dic = convert_holistic_to_dict(processed_frame["holistic"])
    stand, distance = center_and_scale_from_raw(gesture_dic, gpdict,moving_average)
    return stand, distance




def filter_body_part_faster(landmark, ref_list : list):
    if landmark is not None:
        landmark = landmark.landmark
    else:
        return [(np.nan,np.nan,np.nan) for v in ref_list]
    
    return[(landmark[val].x,landmark[val].y,landmark[val].z) for val in ref_list]

def filter_body_parts_faster(landmarks : dict, ref : dict):
    ret = []
    for key, value in ref.items():
        ret.append(filter_body_part_faster(landmarks[key],value))
    return ret

def center_and_scale_from_given(points, gpdict_flattened, moving_average = None): #derived point values and the gesture point dictionary
    if "face" in points.keys() and "pose" in points.keys():
        irises = ([(points["face"].landmark[l].x,points["face"].landmark[l].y,points["face"].landmark[l].z) for l in DICT["scalar"]["face"]["left_iris"]], [(points["face"].landmark[r].x,points["face"].landmark[r].y,points["face"].landmark[r].z) for r in DICT["scalar"]["face"]["right_iris"]])

        dist = distance(np.mean(irises[0],axis=0),np.mean(irises[1],axis=0))

        center = np.mean([(points["pose"].landmark[c].x,points["pose"].landmark[c].y,points["pose"].landmark[c].z) for c in DICT["center"]["pose"]["chest"]],axis = 0)
        curr_scale = dist/SCALE
    else:
        print("no face and/or pose")
        return None
    if moving_average is not None:
        set_scale = np.mean((curr_scale,np.mean(moving_average)))
    else:
        set_scale = curr_scale
    ret = []
    s = time.time()
    for key, value in gpdict_flattened.items():
        if points[key] == None:
            a = np.empty((len(value),3)); a.fill(np.nan)
            ret.append(a)
        else:
            a = points[key].landmark
            ret.append([((a[val].x,a[val].y,a[val].z)-center)/curr_scale for val in value])
    return ret,curr_scale
