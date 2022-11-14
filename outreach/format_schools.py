import pandas as pd
import numpy as np
import os
import csv
#takes directory of text files and formats them into a csv list of teachers for each school, then reorganizes to have each school be their own folder
from tkinter import filedialog as fd
def file_selector(root):
    while True:
        filename = fd.askopenfilenames(title = "select files", initialdir = root) #,
        if filename:
            break
    return filename
def folder_selector(root):
        while True:
            filename = fd.askdirectory(title = "select a folder", initialdir = root) #,
            if filename:
                break
        return filename
def reformat(file):
    f = open(file, "r")
    a = f.read()
    a = a.replace("|", ",")
    remove_space = a
    remove_space = ",".join(remove_space.split(" , "))
    return remove_space.split("\n")
def csv_write(rows, savename):
    with open(savename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in rows:
            i = i.split(",")
            writer.writerow(i)    
def files(specific_files : bool = False):
    if not specific_files:
        base_path = folder_selector(os.path.join(os.path.dirname(os.path.realpath(__file__))))
        school_paths = [d for d in os.listdir(base_path) if ".txt" in d]
        return school_paths, base_path
    else:
        school_paths = file_selector(os.path.join(os.path.dirname(os.path.realpath(__file__))))
        base_path = os.path.dirname(school_paths[0])
        return school_paths, base_path
school_paths,base_path = files(True)
for school in school_paths:
    dir_path = os.path.join(base_path, school)
    formatted_directory = reformat(dir_path)
    os.makedirs(os.path.splitext(dir_path)[0])
    os.rename(dir_path,os.path.join(base_path,os.path.splitext(school)[0],school))
    save_path = os.path.join(os.path.splitext(dir_path)[0],os.path.splitext(school)[0] + ".csv")
    csv_write(formatted_directory, save_path)