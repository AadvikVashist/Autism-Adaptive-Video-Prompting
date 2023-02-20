
import json
import numpy as np
import noduro_code.read_settings as read_settings
DEFAULT_FILE,_ = read_settings.get_settings()
DEFAULT_FILE = DEFAULT_FILE["filesystem"]["gesture_points"] #in settings
from noduro import read_json, flatten

def get_face_dict(file = DEFAULT_FILE):
    file = read_json(file)
    return file["face"]
def get_face_indexes(file = DEFAULT_FILE):
    dict = get_face_dict(file)
    indexes = flatten(dict)
    return indexes


def get_none_dict(file = DEFAULT_FILE):
    file = read_json(file)
    file = file["no_list"]
    return file
def get_none_indexes(file = DEFAULT_FILE):
    file = read_json(file)
    file = file["no_list"]
    return flatten(file)


def get_pose_dict(file = DEFAULT_FILE):
    file = read_json(file)
    return file["pose"]
def get_pose_indexes(file = DEFAULT_FILE):
    dict = get_pose_dict(file)
    indexes = flatten(dict)
    return indexes


def get_hand_dict(file = DEFAULT_FILE):
    file = read_json(file)
    return file["hand"]
def get_hand_indexes(file = DEFAULT_FILE):
    dict = get_hand_dict(file)
    indexes = flatten(dict)
    return indexes

def fix_file(file = DEFAULT_FILE):
    list_dict = get_face_indexes(file)
    bad_dict = get_none_dict(file)
    bad_list = get_none_indexes(file)
    seen = list(set(bad_list))
    uniq = []
    for x in bad_list:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    print(uniq)
    new_dict = {}
    for key, value in bad_dict.items():
        lis = []
        for index in value:
            if index in list_dict:
                x = 0
            else:
                lis.append(index)
        new_dict[key] = lis
    with open("sample.json", "w") as outfile:
        json.dump(new_dict, outfile)

