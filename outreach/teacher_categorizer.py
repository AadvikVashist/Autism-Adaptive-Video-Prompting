import smtplib
import yagmail
import os
import json
import tkinter as tk
from tkinter import filedialog as fd
import pandas as pd
import csv
from pathlib import Path
import numpy as np
import re
from thefuzz import fuzz
__location__= os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
def validNumber(phone_number):
        pattern = re.compile("^[\dA-Z]{3}-[\dA-Z]{3}-[\dA-Z]{4}$", re.IGNORECASE)
        return pattern.match(phone_number) is not None
def format_job(listed : list, default_value : str = "educator" , job_index = 1, unaccepted_values = ["@"]):
    ret = []
    for i in listed:
        for unaccepted_value in unaccepted_values:
            if unaccepted_value in i[job_index]:
                i.insert(job_index, default_value)
                break
        ret.append(i)
    return ret
def remove_website(listed : list, remove_value : list = ["website"], phone_number = False, email = True):
    ret = []
    for i in listed:
        a = []
        if email:
            emails = False
            for val in i:
                if check_email(val):
                    emails = True
            if not emails:
                print("skipping", i, "because no email address")
                continue
        for val in i:
            format = val.strip()
            format = format.lower()
            true = True
            if not phone_number and validNumber(format):
                true = False       
            for remove_val in remove_value:
                if remove_val in format:
                    true = False
            if true:
                a.append(val)
        ret.append(a)
    return ret
def check_email(email):
        regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        # pass the regular expression
        # and the string into the fullmatch() method
        return (re.fullmatch(regex, email)) 
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
    hierarchy_text = [every for every in everything if "hierarchy.txt" in every]
    everything = [every for every in everything if "hierarchy.txt" not in every] 
    arr = []
    for every in everything:
        if every.endswith(filetype):
            lines=list(csv.reader(open(every)))
            lines =list(filter(None, lines))
            curr_folder = os.path.relpath(every,folder)
            curr_folder = os.path.split(curr_folder)[0]
            curr_folder = Path(curr_folder).parts
            curr_folder = [f.replace("_", " ") for f in curr_folder]
            for line in lines:
                line.extend(curr_folder)
                for index in range(len(line)):
                    line[index] = re.sub(r'\t', '', line[index])
            lines = remove_website(lines, ["website", "coach"])
            lines = format_job(lines)
            arr.extend(lines)
    return arr, hierarchy_text
def compare_two_words(a : str, b : str, ratio : float):
    a.lower(); b.lower()
    if b in a:
        return True
    ration = fuzz.ratio(a, b)/100
    return ration >= ratio
def get_matched_values(filter_value, lists, match_percentage):
    ret = []
    if type(filter_value) == list:
        for filter in filter_value:
            for datum in lists:
                percentages =[compare_two_words(datu, filter, match_percentage) for datu in datum]
                if any(percentages):
                    ret.append(datum)
                x = 0
    else:
        filter = filter_value
        for datum in lists:
                percentages =[compare_two_words(datu, filter, match_percentage) for datu in datum]
                if any(percentages):
                    ret.append(datum)
                x = 0
    return ret
def categorization(data_values : list, filter : list, match_percentage :  float = 0.70):
    return_values = []
    # if "ITL" in filter:
    #     ITL = [datum for datum in data_values if any([True if ((("ITL" in datu) or ("team leader" in datu.lower())) and ("@" not in datu)) else False for index,datu in enumerate(datum[1::])])]
    #     return_values.extend(ITL)
    #     filter.remove("ITL")
    for accepted in filter:
        returned = get_matched_values(accepted, data_values,match_percentage)
        # return_values.extend(value)
        return_values.extend(returned)
    for value in return_values:
        print(value)
    return return_values