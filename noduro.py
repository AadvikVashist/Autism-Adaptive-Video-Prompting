import json
import os
import platform
from tkinter import filedialog as fd
import tkinter as tk
import numpy as np
import csv
import pandas as pd
def get_root():
    return os.path.dirname(__file__.replace("\\","/"))
def subdir_path(*args):
    return join(get_root(), *args)
def join(*args):
    return os.path.join(*args).replace("\\","/")

def write_csv(file: str, data, write_or_add : bool = False, relative_path : bool = False, rows_or_row = None):
    if ".csv" not in file:
        raise ValueError(".csv not in file : '" + file+ "'")
    elif relative_path:
        file = subdir_path(file)
    else:
        file = join(file)
    
    if write_or_add: mod = "w" 
    else: mod = "a"
    with open(file, mode =  mod, newline = "") as csv_file:
        obj = csv.writer(csv_file)
        if rows_or_row is None:

            if len(np.asarray(data, dtype=object).shape) < 2:
                obj.writerow(data)
            else:
                obj.writerows(data)
        else:
            if rows_or_row:
                obj.writerows(data)
            else:
                obj.writerow(data)
        csv_file

def read_csv(file: str,relative_path : bool = False):
    if ".csv" not in file:
        raise ValueError(".csv not in file: '" + file+ "'")
    elif relative_path:
        file = subdir_path(file)
    else:
        file = join(file)
    if os.path.exists(file) == False: raise ValueError(file + " does not exist") 
    return np.asarray(pd.read_csv(file, header = None))

def write_json(file : str, data : dict, relative_path = False, write_or_add : bool = False):
    if ".json" not in file:
        raise ValueError(".json not in file : '" + file+ "'")
    elif relative_path:
        file = subdir_path(file)
    else:
        file = join(file)
    if write_or_add:
        json_object = data
    else: 
        try:
            json_object = read_json(file)
            json_object.update(data)
        except:
            json_object = data
    json_object = json.dumps(json_object, indent=4)

    # Writing to sample.json
    with open(file, "w") as outfile:
        outfile.write(json_object)

def read_json(file, relative_path = False, write_if_doesnt_exist = False):
    if ".json" not in file:
        raise ValueError(".json not in file: '" + file+ "'")
    elif relative_path:
        file = subdir_path(file)
    else:
        file = join(file)
    if os.path.exists(file) == False and write_if_doesnt_exist == False: raise ValueError(file + " does not exist")
    elif os.path.exists(file) == False and write_if_doesnt_exist == True: 
        write_json(file, {}, True, True)
    file_data = dict(json.load(open(file)))
    return file_data

def folder_selector(root = None):
    if not root:
        if platform.system() == 'Windows': # this hasn't been tested 
            desktop = join(join(os.environ['USERPROFILE']), 'Desktop') #windows 
        elif platform.system() == 'Darwin':
            desktop = join(join(os.path.expanduser('~')), 'Desktop') #mac 
        elif platform.system() == 'Linux':
            desktop = join(join(os.path.expanduser('~')), 'Desktop') #linux
        else:
            raise Exception("Unsupported operating system: " + platform.system())
    while True:
        filename = fd.askdirectory(title = "select a folder", initialdir = desktop)
        if filename is not None:
            break
    return filename

def file_selector(root = None, file_type = None):
    if root is None:
        if platform.system() == 'Windows': # this hasn't been tested 
            desktop = join(os.environ['USERPROFILE'], 'Desktop') #windows 
        elif platform.system() == 'Darwin':
            desktop = join(os.path.join(os.path.expanduser('~')), 'Desktop') #mac 
        elif platform.system() == 'Linux':
            desktop = join(os.path.join(os.path.expanduser('~')), 'Desktop') #linux
        else:
            raise Exception("Unsupported operating system: " + platform.system())
    else:
        desktop = root
    if file_type is None:
        filename = fd.askopenfilename(title = "select a file", initialdir = desktop)
    else:
        filename = fd.askopenfilename(title = "select a file", initialdir = desktop, filetypes= file_type)
    return filename

def flatten(d):
    ret = []
    for i in d.values():
        try:
            ret.extend(i)
        except:
            ret.append(i)
    return ret
def get_dir_files(path: str, relative_path : bool = False,return_path_relative_to_base : bool = False):
    if relative_path:
        path = subdir_path(path) 
    else:
        path = join(path)
    if return_path_relative_to_base:

        result = [join(os.path.relpath(dp, path), f) for dp, dn, filenames in os.walk(path) for f in filenames if "pycache" not in dp and "git" not in dp]
    else:
        result = [join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if "pycache" not in dp and "git" not in dp]

    return result

def check_boolean_input(input : str) -> bool:
    input = input.lower()
    if "yes" in input or "true" in input:
        return True
    elif "no" in input or "false" in input:
        return False
    if ("y" in input or "t" in input) and ("n" in input or "f" in input):
        raise ValueError("yes or no not in input. Please try again")
    elif ("y" in input or "t" in input) and len(input) < 4:
        return True
    elif ("n" in input or "f" in input) and len(input) < 4:
        return False
    else:
        raise ValueError("yes or no not in input. Please try again")
def make_dir_if_not_exist(file : str, relative_path = False):
    try:
        os.makedirs(file)
        print("made directory '"+file+"'")
    except FileExistsError:
        print("directory", file, "already exists")
        pass
if __name__ == '__main__':
    print("testing get_root:", get_root())
    print("testing subdir_path with desktop\\air\\hi/hi/wasd/how.py:", subdir_path("desktop\\air\\hi/hi/wasd/how.py"))
    print("testing flatten:", flatten({0 : "sdf",1  : "Sdf",2 : "sdfs",3 : "21321", 5: [101,1,2,23,12,31,23,123,12]}))
    get_dir_files("",True)
    print("testing file selector:", file_selector())
    print("testing file selector:", folder_selector())
