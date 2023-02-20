import json
import os
import noduro
settings_folder = noduro.subdir_path('settings')
settings_folder_files = [noduro.join(settings_folder, file) for file in os.listdir(settings_folder)]
email_file = list(filter(lambda x : "email" in x, settings_folder_files))[0]
settings_file = list(filter(lambda x : "user_settings" in x, settings_folder_files))[0]

def get_settings(loc = settings_file):
    dict = noduro.read_json(loc)
    return dict, loc 
def get_sequence(loc = settings_file):
    setting, *_ = get_settings(loc)
    return setting["sequence_num"]
def increase_sequence(loc = settings_file):
    with open(loc, "r+") as json_file:
        json_obj = json.load(json_file)
        json_obj["sequence_num"] += 1
        json_file.seek(0)
        json.dump(json_obj, json_file, indent=4)
        # json_file.truncate()

def get_and_increment_sequence(loc = settings_file):
    seq = get_sequence(loc)
    increase_sequence(loc)
    return seq

def set_points(skip, loc = settings_file):
    with open(loc, "r+") as json_file:
        json_obj = json.load(json_file)
        json_obj["frame_skippage"] = skip
        json_file.seek(0)
        json.dump(json_obj, json_file, indent=4)
        # json_file.truncate()
def get_points(loc = settings_file):
    setting, *_ = get_settings(loc)
    return setting["frame_skippage"]

def set_scale_factor(factor, loc = settings_file):
    with open(loc, "r+") as json_file:
        json_obj = json.load(json_file)
        json_obj["scale_factor"] = factor
        json_file.seek(0)
        json.dump(json_obj, json_file, indent=4)
        # json_file.truncate()
def get_scale_factor(loc = settings_file):
    setting, *_ = get_settings(loc)
    return setting["scale_factor"]