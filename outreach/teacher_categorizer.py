import smtplib
import yagmail
import os
import json
import tkinter as tk
from tkinter import filedialog as fd
import pandas as pd
import csv
from pathlib import Path

__location__= os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
def folder_selector(root):
    window = tk.Tk()
    window.wm_attributes('-topmost', 1)
    window.withdraw()
    while True:
        filename = fd.askdirectory(title = "select a folder", initialdir = root) #,
        if filename:
            break
    return filename
def folder_to_list(folder = None, filetype = ".csv"):
    if not folder: folder = folder_selector(__location__)
    everything = [os.path.join(dp, f) for dp, dn, fn in os.walk(folder) for f in fn]
    arr = []
    for every in everything:
        if every.endswith(filetype):
            lines=list(csv.reader(open(every)))
            lines =list(filter(None, lines))
            curr_folder = os.path.relpath(every,folder)
            curr_folder = os.path.split(curr_folder)[0]
            curr_folder = Path(curr_folder).parts
            curr_folder = [f.replace("_", " ") for f in curr_folder]
            curr_folder = '-'.join(curr_folder)
            for line in lines:
                line.append(curr_folder)
            arr.extend(lines)
    return arr
def categorization(data_values : list, filter : list):
    for datum in data_values:
        print(datum)