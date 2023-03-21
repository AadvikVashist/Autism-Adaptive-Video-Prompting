import cv2
import noduro
path = "C:/Users/aadvi/Desktop/Autism/Autism-Adaptive-Video-Prompting/modeling/gesture/DALLE.png"
frame = cv2.imread(path)
image = frame.copy()
boxes = []
while True:
    a = cv2.selectROI("selector", image, showCrosshair = True)
    cv2.rectangle(image, (int(a[0]), int(a[1])), (int(a[0]+a[2]), int(a[1]+a[3])), (200,200,200), -1)
    boxes.append(a)
    cv2.destroyAllWindows()
    inp = input("do you want to make another rectangle")
    if noduro.check_boolean_input(inp):
        continue
    else:
        break