import json
import os
cwd = os.getcwd()
settings_folder = os.path.join(cwd, 'settings')
settings_folder_files = [os.path.join(settings_folder, file) for file in os.listdir(settings_folder)]
email_file = list(filter(lambda x : "email" in x, settings_folder_files))[0]
settings_file = list(filter(lambda x : "user_settings" in x, settings_folder_files))[0]

def get_settings(loc = settings_file):
    json_obj = json.load(open(loc))
    data = dict(json_obj)
    return data, json_obj, loc 
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