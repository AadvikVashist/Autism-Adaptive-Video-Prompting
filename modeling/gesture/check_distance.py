import noduro_code.read_settings as read_settings
import modeling.gesture.pose_manipulation.pose_standardizer as standardizer
try:
    SCALE = read_settings.get_scale_factor()
except:
    print("no scale")
MAXIMUM_DIFFERENCE = 0.2
MIN = SCALE*(1-MAXIMUM_DIFFERENCE)
MAX = SCALE*(1+MAXIMUM_DIFFERENCE)

def closer_or_farther(points):
    dist = standardizer.calibrate_pose_using_eye(points)
    if dist < MIN:
        return "move closer"
    elif dist > MAX:
        return "move farther"
    else:
        return "perfect distance"