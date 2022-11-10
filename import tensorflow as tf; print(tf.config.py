import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))
   
    # def draw_hands(self, image, mediapipehands, handslms = None):
    #     if not handslms:
    #         handslms = get_hands(image, mediapipehands, mediaPipeHandModel)
    #     mpDraw.draw_landmarks(image, handslms, mediaPipeHandSolutions.HAND_CONNECTIONS)

    # def get_hands(self, image, mediapipehandmodel, hands):
    #     result = mediapipehandmodel.process(image)
    #     if result.multi_hand_landmarks:
    #         landmarks = []
    #         for handslms in result.multi_hand_landmarks:
    #             for lm in handslms.landmark:
    #                 # print(id, lm)
    #                 lmx = int(lm.x * x)
    #                 lmy = int(lm.y * y)
    #                 landmarks.append([lmx, lmy])
    #     return landmarks, hands
# # mpFace = mp.solutions.face_detection
# # face_detection = mpFace.FaceDetection(model_selection=0, min_detection_confidence=0.5)
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# mpDraw = mp.solutions.drawing_utils

# # Load the gesture recognizer model
# model = load_model('mp_hand_gesture')

# # Load class names
# f = open('gesture.names', 'r')
# classNames = f.read().split('\n')
# f.close()
# # print(classNames)


# # Initialize the webcam
# cap = cv2.VideoCapture(0)


# while True:
#     # Read each frame from the webcam
#     _, frame = cap.read()

#     x, y, c = frame.shape

#     # Flip the frame vertically
#    
#     framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Get hand landmark prediction
#     draw_hands(framergb,)
#     faces = detector(gray)
#     # print(result)
#     for face in faces:
#         x1 = face.left() # left point
#         y1 = face.top() # top point
#         x2 = face.right() # right point
#         y2 = face.bottom() # bottom point

#         # Look for the landmarks
#         landmarks = predictor(image=gray, box=face)
#         for n in range(0, 68):
#             x = landmarks.part(n).x
#             y = landmarks.part(n).y

#             # Draw a circle
#             cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

#     className = ''
#     # frame.fill(0)
#     # post process the result


#             # Drawing landmarks on frames

#             # Predict gesture
#             # prediction = model.predict([landmarks])
#             # # print(prediction)
#             # classID = np.argmax(prediction)
#             # className = classNames[classID]
#     # Show the final output
#     cv2.imshow("Output", frame) 

#     if cv2.waitKey(1) == ord('q'):
#         break

# # release the webcam and destroy all active windows
# cap.release()

# cv2.destroyAllWindows()

# # For static images:
# IMAGE_FILES = []
# BG_COLOR = (192, 192, 192) # gray
# with mp_pose.Pose(
#     static_image_mode=True,
#     model_complexity=2,
#     enable_segmentation=True,
#     min_detection_confidence=0.5) as pose:
#     for idx, file in enumerate(IMAGE_FILES):
#         image = cv2.imread(file)
#         image_height, image_width, _ = image.shape
#         # Convert the BGR image to RGB before processing.
#         results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#         if not results.pose_landmarks:
#             continue
#         print(
#             f'Nose coordinates: ('
#             f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
#             f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
#         )

#     annotated_image = image.copy()
#     # Draw segmentation on the image.
#     # To improve segmentation around boundaries, consider applying a joint
#     # bilateral filter to "results.segmentation_mask" with "image".
#     condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
#     bg_image = np.zeros(image.shape, dtype=np.uint8)
#     bg_image[:] = BG_COLOR
#     annotated_image = np.where(condition, annotated_image, bg_image)
#     # Draw pose landmarks on the image.
#     mp_drawing.draw_landmarks(
#         annotated_image,
#         results.pose_landmarks,
#         mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#     cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
#     # Plot pose world landmarks.
#     mp_drawing.plot_landmarks(
#         results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# # For webcam input:
# cap = cv2.VideoCapture(0)
# with mp_pose.Pose(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as pose:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue

#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = pose.process(image)

#     # Draw the pose annotation on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     mp_drawing.draw_landmarks(
#         image,
#         results.pose_landmarks,
#         mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()