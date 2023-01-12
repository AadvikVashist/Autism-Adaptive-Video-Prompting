import cv2
import mediapipe as mp
import os
import platform
from tkinter import filedialog as fd
import tkinter as tk
import time
import math
import matplotlib.cm 
import mediapipe.python.solutions.face_mesh_connections as mp_eyes
import matplotlib.pyplot as plt
from gesture_tracker    import gesture_tracker
import math
import numpy as np
from numpy import linalg
class eye_tracking(gesture_tracker):
    def perpendicular_bisector(self, a,b):
        vx = b[0] - a[0]
        vy = b[1] - a[1]
        mag = np.sqrt(vx**2 + vy**2)
        vx = vx/mag
        vy = vy/mag
        vt = vx
        vx = -vy
        vy = vt
        return vx,vy
    def pythag(self,a,b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    def distance_between_line_and_point(self, p1,p2,p3):
        d= (p3[0]-p1[0])*(p2[1]-p1[1])-(p3[1]-p1[1])*(p2[0]-p1[0])
        if d > 0:
            d = 1
        else:
            d = -1
        return linalg.norm(np.cross(p2-p1, p1-p3))/linalg.norm(p2-p1)*d
    def nose_line_angle(self,list):
        fit = np.polyfit(list[:,0],list[:,1],1)
        return fit, -1*np.degrees(np.arctan(1/fit[0]))
    def angle_between_two_lines(self,M1,M2):
        angle = abs((M2 - M1) / (1 + M1 * M2))
        return np.degrees(np.arctan(angle))


    def eyes(self, frame, pointed):
        face_list = pointed.face_landmarks
        pose_list = pointed.pose_landmarks
        
        
        right_iris = list(set(self.gesture_point_dict["face"]["right_iris"]))
        left_iris = list(set(self.gesture_point_dict["face"]["left_iris"]))
        chest = list(set(self.gesture_point_dict["pose"]["chest"]))
        nose_line = list(set(self.gesture_point_dict["face"]["nose_line"]))

        image_rows, image_cols, _ = frame.shape
        
        if any((right_iris is None, left_iris is None)):
            return
        
       
        right_iris_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(face_list.landmark) if index in right_iris])
        left_iris_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(face_list.landmark) if index in left_iris])
        chest_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(pose_list.landmark) if index in chest])

        nose_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(face_list.landmark) if index in nose_line])


        right_box = cv2.boundingRect(right_iris_list)
        left_box = cv2.boundingRect(left_iris_list)

        eye_center = np.asarray([int(np.mean((right_box[0], left_box[0] + left_box[2]))),int(np.mean((right_box[1], left_box[1] + left_box[3])))])
        body_center = np.mean(chest_list, axis = 0, dtype = np.int16)
        body_slope = self.perpendicular_bisector(chest_list[0],chest_list[1])
        new_point = np.asarray([int(body_center[0]+body_slope[0]*100),int(body_center[1]+body_slope[1]*100)])
        
        eye_distance = self.distance_between_line_and_point(new_point, body_center, eye_center)
        chest_distance = self.pythag(chest_list[0],chest_list[1])/2
        king_joshua_ratio = np.round(eye_distance/chest_distance,3)

        nose_line, nose_angle = self.nose_line_angle(nose_list)
        nose_line = nose_line[0]

        angle_diff = self.angle_between_two_lines(-1/nose_line, body_slope[1]/body_slope[0])
        cv2.putText(frame, "Orientation: " + str(nose_angle), (10,45), cv2.FONT_HERSHEY_PLAIN, fontScale = 1, thickness= 2, color = (0,0,0))
        cv2.putText(frame, "Diff Ratio: " + str(angle_diff), (10,60), cv2.FONT_HERSHEY_PLAIN, fontScale = 1, thickness= 2, color = (0,0,0))
        cv2.putText(frame, "Eye Ratio: " + str(king_joshua_ratio), (10,30), cv2.FONT_HERSHEY_PLAIN, fontScale = 1, thickness= 2, color = (0,0,0))
        cv2.circle(frame, eye_center,radius = 2,color = (255,255,255), thickness = 5)
        cv2.rectangle(frame,(right_box[0],right_box[1]),(right_box[0] + right_box[2],right_box[1] +right_box[3]), color = (255,255,255))
        cv2.rectangle(frame,(left_box[0],left_box[1]),(left_box[0] + left_box[2],left_box[1] +left_box[3]), color = (255,255,255))
        cv2.line(frame,chest_list[0],chest_list[1], color = (0,0,255),thickness = 2)
        cv2.line(frame,body_center,new_point, color = (0,0,255),thickness = 2)
        cv2.line(frame,(int((king_joshua_ratio)*image_cols/2+image_cols/2), 0), (int((king_joshua_ratio)*image_cols/2+image_cols/2), 10000), color = (0,0,255),thickness = 3)
        # cv2.imshow("frame",frame)
        # cv2.waitKey(1)
        return frame
#         cv2.fillPoly(mask, pts = [right_eye_list], color = (255,255,255))
#         cv2.fillPoly(mask, pts = [left_eye_list], color = (255,255,255))
#         right_eye_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(landmark_list.landmark) if index in right_eye])
#         left_eye_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(landmark_list.landmark) if index in left_eye])
        

#         kernel = np.ones((2,2),np.uint8)
#         mask = cv2.dilate(mask,kernel,iterations = 5)
#         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#         contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         x,y,w,h = cv2.boundingRect(np.concatenate(contours))
#         mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#         masked_img = cv2.bitwise_and(frame, mask)[y:y+h,x:x+w]
#         while masked_img.shape[0] < self.optimal_y:
#             scalar = round(masked_img.shape[0] / self.optimal_y)
#             if scalar >= 4:
#                 masked_img = self.super_resolution4.upsample(masked_img)
#             elif scalar >= 3:
#                 masked_img = self.super_resolution3.upsample(masked_img)
#             else:
#                 masked_img = self.super_resolution2.upsample(masked_img)
#         # cv2.resize(masked_img, (int(masked_img.shape[1]*6),int(masked_img.shape[0]*6)), interpolation = cv2.INTER_AREA)
#         cv2.imshow('Masked Image Upscale', masked_img)
#         # contours, hierarchy = cv2.findContours(cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         # cv2.drawContours(masked_img, contours, -1, (0, 255, 0), 3)
#         # # masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
#         # # masked_img = cv2.resize(masked_img, (int(masked_img.shape[1]*6),int(masked_img.shape[0]*6)), interpolation = cv2.INTER_AREA)
#         # cv2.imshow('Masked Image', masked_img)
#         # gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
#         # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
#         # # ensure at least some circles were found
#         # if circles is not None:
#         #     # convert the (x, y) coordinates and radius of the circles to integers
#         #     circles = np.round(circles[0, :]).astype("int")
#         #     # loop over the (x, y) coordinates and radius of the circles
#         #     for (x, y, r) in circles:
#         #         # draw the circle in the output image, then draw a rectangle
#         #         # corresponding to the center of the circle
#         #         cv2.circle(masked_img, (x, y), r, (0, 255, 0), 4)
#         # cv2.imshow('Masked Image', masked_img)
#         # cv2.waitKey(0)
# # Width of eyes in relation to each other
# # Centering of eyes relative to midpoint of shoulders. Left-right based on the center
a = eye_tracking()
a.realtime_analysis()
# a.video_analysis("C:/Users/aadvi/Desktop/Movie on 1-11-23 at 10.00 AM.mov")