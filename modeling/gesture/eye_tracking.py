import cv2
import numpy as np
from numpy import linalg
from numpy.polynomial import Polynomial as P
import warnings
def perpendicular_bisector(a,b):
    vx = b[0] - a[0]
    vy = b[1] - a[1]
    mag = np.sqrt(vx**2 + vy**2)
    vx = vx/mag
    vy = vy/mag
    vt = vx
    vx = -vy
    vy = vt
    return vx,vy

def pythag(a,b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def distance_between_line_and_point( p1,p2,p3):
    d= (p3[0]-p1[0])*(p2[1]-p1[1])-(p3[1]-p1[1])*(p2[0]-p1[0])
    if d > 0:
        d = 1
    else:
        d = -1
    return linalg.norm(np.cross(p2-p1, p1-p3))/linalg.norm(p2-p1)*d

def nose_line_angle(list):
    with warnings.catch_warnings():
        try:
            fit = np.polyfit(list[:,0],list[:,1],1)
        except:
            fit = [100000,0]
        return fit, -1*np.degrees(np.arctan(1/fit[0]))

def angle_between_two_lines(M1,M11,M2,M22):
    if M11 == 0:
        Mx = 0
    else:
        Mx = M1/M11
    if M22 == 0:
        My = 0
    else:
        My = M2/M22
    
    if M1*M2 == -1:
        return 90
    elif abs(M1-M2) < 0.001:
        return 0
    else:
        angle = abs((M2 - M1) / (1 + M1 * M2))
        return np.degrees(np.arctan(angle))

def landmarks_to_lists( landmarks):
    return landmarks.face_landmarks, landmarks.pose_landmarks

def calculate_eye_ratio( left_iris_list, right_iris_list, chest_list, nose_list):
    right_box = cv2.boundingRect(right_iris_list)
    left_box = cv2.boundingRect(left_iris_list)

    eye_center = np.asarray([int(np.mean((right_box[0], left_box[0] + left_box[2]))),int(np.mean((right_box[1], left_box[1] + left_box[3])))])
    body_center = np.mean(chest_list, axis = 0, dtype = np.int16)
    body_slope = perpendicular_bisector(chest_list[0],chest_list[1])
    factor = (eye_center[1]-body_center[1])/body_slope[1]*2
    new_point = np.asarray([int(body_center[0]+body_slope[0]*factor),int(body_center[1]+body_slope[1]*factor)])
    
    eye_distance = distance_between_line_and_point(new_point, body_center, eye_center)
    chest_distance = pythag(chest_list[0],chest_list[1])/2
    king_joshua_ratio = np.round(eye_distance/chest_distance,3)

    nose_line, nose_angle = nose_line_angle(nose_list)
    nose_line = nose_line[0]
    if nose_line == 0:
        print("h")
    angle_diff = angle_between_two_lines(-1,nose_line, body_slope[1],body_slope[0])
    return king_joshua_ratio, nose_angle,angle_diff, new_point, body_center, eye_center, left_box, right_box

def draw_eye_calculations(frame,eye_center,angle_diff,king_joshua_ratio, body_center, nose_angle, chest_list, new_point):  
    cv2.putText(frame, "Orientation: " + str(nose_angle), (10,45), cv2.FONT_HERSHEY_PLAIN, fontScale = 1, thickness= 2, color = (0,0,0))
    cv2.putText(frame, "Angle Diff: " + str(angle_diff), (10,60), cv2.FONT_HERSHEY_PLAIN, fontScale = 1, thickness= 2, color = (0,0,0))
    cv2.putText(frame, "Eye Ratio: " + str(king_joshua_ratio), (10,30), cv2.FONT_HERSHEY_PLAIN, fontScale = 1, thickness= 2, color = (0,0,0))
    
    cv2.circle(frame, eye_center,radius = 2,color = (255,255,255), thickness = 2) #dot in the center of two eyes
    
    
    cv2.line(frame,chest_list[0],chest_list[1], color = (102,35,0),thickness = 2) #  Connect two shoulders
    cv2.line(frame,body_center,new_point, color = (102,35,0),thickness = 2) # Perpendicular Bisector
    # cv2.line(frame,(int((king_joshua_ratio)*image_cols/2+image_cols/2), 0), (int((king_joshua_ratio)*image_cols/2+image_cols/2), 10000), color = (0,0,255),thickness = 3) # moving line

    return frame