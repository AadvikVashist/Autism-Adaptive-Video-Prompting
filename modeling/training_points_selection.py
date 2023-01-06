from gesture_tracker import gesture_tracker
import cv2
import tracking_points as points
import matplotlib.cm as cm
import numpy as np
class training_points_selection_fresh_start(gesture_tracker):
    def draw_face_proprietary(self, frame, landmark_list):
            self.pre_resize = 3
            self.resize = 2
            image_rows, image_cols, _ = frame.shape
            cmap = cm.get_cmap('hsv')
            frame = cv2.resize(frame, [frame.shape[1]*self.pre_resize, frame.shape[0]*self.pre_resize])
            lister = []
            for idx, landmark in enumerate(landmark_list.landmark):
                landmark_px = self._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                            image_cols, image_rows)
                cv2.circle(frame, [landmark_px[0]*3, landmark_px[1]*3], 2, np.array(cmap(0.5))*255, 1)
                cv2.putText(frame, str(idx),  [landmark_px[0]*3, landmark_px[1]*3], fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4, color =  np.array(cmap(1))*255, thickness = 1, lineType = 2)
            lister = np.array(lister)
            mins = np.min(lister, axis = 0)
            maxs = np.max(lister, axis = 0)
            frame = frame[mins[1]*3-50:maxs[1]*3+50, mins[0]*3-50:maxs[0]*3+50]
            cv2.imshow("frame", cv2.resize(frame, [int(frame.shape[1]*self.resize), int(frame.shape[0]*self.resize)]))
            return frame, idx


class training_points_selection_fresh_start_no_cropping(gesture_tracker):
    def draw_face_proprietary(self, frame, landmark_list):
            self.pre_resize = 3
            self.resize = 2
            image_rows, image_cols, _ = frame.shape
            cmap = cm.get_cmap('hsv')
            frame = cv2.resize(frame, [frame.shape[1]*self.pre_resize, frame.shape[0]*self.pre_resize])
            lister = []
            for idx, landmark in enumerate(landmark_list.landmark):
                landmark_px = self._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                            image_cols, image_rows)
                cv2.circle(frame, [landmark_px[0]*3, landmark_px[1]*3], 2, np.array(cmap(0.5))*255, 1)
                cv2.putText(frame, str(idx),  [landmark_px[0]*3, landmark_px[1]*3], fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4, color =  np.array(cmap(1))*255, thickness = 1, lineType = 2)
            lister = np.array(lister)
            mins = np.min(lister, axis = 0)
            maxs = np.max(lister, axis = 0)
            cv2.imshow("frame", cv2.resize(frame, [int(frame.shape[1]*self.resize), int(frame.shape[0]*self.resize)]))
            return frame, idx

class training_points_selection_using_existing(gesture_tracker):
    def draw_face_proprietary(self, frame, landmark_list):
            self.pre_resize = 3
            self.resize = 2
            self.pose_dict = points.get_pose_dict()
            self.face_dict = points.get_face_dict()
            self.none = points.get_none()
            image_rows, image_cols, _ = frame.shape
            cmap = cm.get_cmap('hsv')
            face_dict = []
            for key, dictionary in self.face_dict.items():
                if dictionary is not None:
                    face_dict.extend(dictionary)
            frame = cv2.resize(frame, [frame.shape[1]*self.pre_resize, frame.shape[0]*self.pre_resize])
            lister = []
            for idx, landmark in enumerate(landmark_list.landmark):
                if idx in face_dict:
                    landmark_px = self._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                                image_cols, image_rows)
                    cv2.circle(frame, [landmark_px[0]*3, landmark_px[1]*3], 2, np.array(cmap(0.9))*255, 2)
                    lister.append(landmark_px)
                if idx not in face_dict and idx not in self.none:
                    landmark_px = self._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                                image_cols, image_rows)
                    cv2.circle(frame, [landmark_px[0]*3, landmark_px[1]*3], 2, np.array(cmap(0.5))*255, 1)
                    cv2.putText(frame, str(idx),  [landmark_px[0]*3, landmark_px[1]*3], fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4, color =  np.array(cmap(1))*255, thickness = 1, lineType = 2)
                    print(idx)
                    
            lister = np.array(lister)
            mins = np.min(lister, axis = 0)
            maxs = np.max(lister, axis = 0)
            frame = frame[mins[1]*3-50:maxs[1]*3+50, mins[0]*3-50:maxs[0]*3+50]
            cv2.imshow("frame", cv2.resize(frame, [int(frame.shape[1]*self.resize), int(frame.shape[0]*self.resize)]))
            return frame, idx
class training_points_selection_using_existing_no_crop(gesture_tracker):
    def draw_face_proprietary(self, frame, landmark_list):
            self.pre_resize = 3
            self.resize = 2
            self.pose_dict = points.get_pose_dict()
            self.face_dict = points.get_face_dict()
            self.none = points.get_none()
            image_rows, image_cols, _ = frame.shape
            cmap = cm.get_cmap('hsv')
            face_dict = []
            for key, dictionary in self.face_dict.items():
                if dictionary is not None:
                    face_dict.extend(dictionary)
            frame = cv2.resize(frame, [frame.shape[1]*self.pre_resize, frame.shape[0]*self.pre_resize])
            lister = []
            for idx, landmark in enumerate(landmark_list.landmark):
                if idx in face_dict:
                    landmark_px = self._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                                image_cols, image_rows)
                    cv2.circle(frame, [landmark_px[0]*3, landmark_px[1]*3], 2, np.array(cmap(0.9))*255, 2)
                    lister.append(landmark_px)
                if idx not in face_dict and idx not in self.none:
                    landmark_px = self._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                                image_cols, image_rows)
                    cv2.circle(frame, [landmark_px[0]*3, landmark_px[1]*3], 2, np.array(cmap(0.5))*255, 1)
                    cv2.putText(frame, str(idx),  [landmark_px[0]*3, landmark_px[1]*3], fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4, color =  np.array(cmap(1))*255, thickness = 1, lineType = 2)
                    print(idx)
                    
            lister = np.array(lister)
            mins = np.min(lister, axis = 0)
            maxs = np.max(lister, axis = 0)
            frame = frame[mins[1]*3-50:maxs[1]*3+50, mins[0]*3-50:maxs[0]*3+50]
            cv2.imshow("frame", cv2.resize(frame, [int(frame.shape[1]*self.resize), int(frame.shape[0]*self.resize)]))
            return frame, idx