import cv2
import numpy as np
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
import tracking_points as points
from gesture_tracker    import gesture_tracker
class eye_tracking(gesture_tracker):
    def eyes(self, frame, landmark_list):
        mask = np.zeros_like(frame)
        framed = self.face_pose(frame,landmark_list)
        cv2.imshow("framed",framed)
        right_eye = list(set(np.array(list(mp_eyes.FACEMESH_RIGHT_EYE)).flatten()))
        left_eye = list(set(np.array(list(mp_eyes.FACEMESH_LEFT_EYE)).flatten()))
        right_iris = list(set(np.array(list(mp_eyes.FACEMESH_RIGHT_IRIS)).flatten()))
        left_iris = list(set(np.array(list(mp_eyes.FACEMESH_LEFT_IRIS)).flatten()))
        image_rows, image_cols, _ = frame.shape    
        if any((right_eye is None, left_eye is None, right_iris is None, left_iris is None)):
            return
        right_eye_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(landmark_list.landmark) if index in right_eye])
        left_eye_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(landmark_list.landmark) if index in left_eye])
        right_iris_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(landmark_list.landmark) if index in right_iris])
        left_iris_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(landmark_list.landmark) if index in left_iris])
        
        cv2.fillPoly(mask, pts = [right_eye_list], color = (255,255,255))
        cv2.fillPoly(mask, pts = [left_eye_list], color = (255,255,255))
        right_eye_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(landmark_list.landmark) if index in right_eye])
        left_eye_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(landmark_list.landmark) if index in left_eye])
        
        # right_iris_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(landmark_list.landmark) if index in right_iris])
        # left_iris_list = np.array([list(self._normalized_to_pixel_coordinates(land.x, land.y, image_cols, image_rows)) for index, land in enumerate(landmark_list.landmark) if index in left_iris])
        # cv2.fillPoly(mask, pts = [right_iris_list], color = (255,255,255))
        # cv2.fillPoly(mask, pts = [left_iris_list], color = (255,255,255))

        kernel = np.ones((2,2),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations = 5)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x,y,w,h = cv2.boundingRect(np.concatenate(contours))
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        masked_img = cv2.bitwise_and(frame, mask)[y:y+h,x:x+w]
        while masked_img.shape[0] < self.optimal_y:
            scalar = round(masked_img.shape[0] / self.optimal_y)
            if scalar >= 4:
                masked_img = self.super_resolution4.upsample(masked_img)
            elif scalar >= 3:
                masked_img = self.super_resolution3.upsample(masked_img)
            else:
                masked_img = self.super_resolution2.upsample(masked_img)
        # cv2.resize(masked_img, (int(masked_img.shape[1]*6),int(masked_img.shape[0]*6)), interpolation = cv2.INTER_AREA)
        cv2.imshow('Masked Image Upscale', masked_img)
        # contours, hierarchy = cv2.findContours(cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(masked_img, contours, -1, (0, 255, 0), 3)
        # # masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        # # masked_img = cv2.resize(masked_img, (int(masked_img.shape[1]*6),int(masked_img.shape[0]*6)), interpolation = cv2.INTER_AREA)
        # cv2.imshow('Masked Image', masked_img)
        # gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
        # # ensure at least some circles were found
        # if circles is not None:
        #     # convert the (x, y) coordinates and radius of the circles to integers
        #     circles = np.round(circles[0, :]).astype("int")
        #     # loop over the (x, y) coordinates and radius of the circles
        #     for (x, y, r) in circles:
        #         # draw the circle in the output image, then draw a rectangle
        #         # corresponding to the center of the circle
        #         cv2.circle(masked_img, (x, y), r, (0, 255, 0), 4)
        # cv2.imshow('Masked Image', masked_img)
        # cv2.waitKey(0)