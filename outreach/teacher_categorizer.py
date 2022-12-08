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
import csv
def read_csv_file(filepath, dict = True):
    # Open the CSV file
    with open(filepath, 'r') as file:
        # Create a CSV reader
        reader = csv.reader(file)
        if dict:
            # Create a list to hold the dictionaries
            row_dict = {}
            
            # Iterate over the rows in the file
            for index, row in enumerate(reader):
                # Create a dictionary for the row
                rowed = []
                email = False
                # Iterate over the columns in the row
                for i, col in enumerate(row):
                    # Add the column value to the dictionary
                    if "@" in col:
                        email = True
                    rowed.append([ro.strip() for ro in row])
                if email: row_dict["email"] = rowed
                else: row_dict[index] = rowed
                # Return the list of dictionaries
        else:
            row_dict = []
            for row in reader:
                row_dict.extend(row)
            row_dict = [rows.strip() for rows in row_dict]
        return row_dict
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
    blacklist_text = [every for every in everything if "blacklist.txt" in every]
    everything = [every for every in everything if "hierarchy.txt" not in every and "blacklist.txt" not in every and ".csv" in every] 
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
    hierarchy_text = sorted(hierarchy_text, key = len)
    hierarchy_text = read_csv_file(hierarchy_text[0])
    blacklist = []
    for black in blacklist_text:
        blacklist.extend(read_csv_file(black, False))
    return arr, hierarchy_text, blacklist
def compare_two_words(a : str, b : str, ratio : float):
    a.lower(); b.lower()
    if b in a:
        return True
    ration = fuzz.ratio(a, b)/100
    return ration >= ratio
def get_matched_values(filter_value, lists, match_percentage, format = True):
    ret = []
    if type(filter_value) == list:
        for filter in filter_value:
            for datum in lists:
                percentages =[compare_two_words(datu, filter, match_percentage) for datu in datum]
                if any(percentages):
                    datum[percentages.index(True)] = filter_value[0]
                    ret.append(datum)
    else:
        filter = filter_value
        for datum in lists:
            percentages =[compare_two_words(datu, filter, match_percentage) for datu in datum]
            if any(percentages):
                datum[percentages.index(True)] = filter_value[0]
                ret.append(datum)
    return ret
def categorization(data_values : list, filter : list, match_percentage :  float = 0.70, blacklist : list = None):
    return_values = []
    # if "ITL" in filter:
    #     ITL = [datum for datum in data_values if any([True if ((("ITL" in datu) or ("team leader" in datu.lower())) and ("@" not in datu)) else False for index,datu in enumerate(datum[1::])])]
    #     return_values.extend(ITL)
    #     filter.remove("ITL")
    for accepted in filter:
        returned = get_matched_values(accepted, data_values,match_percentage)
        returned = [ret for ret in returned if ret not in return_values]
        # return_values.extend(value)
        return_values.extend(returned)
    if blacklist is not None:
        index = 0
        while index < len(return_values):
            return_value = [i for i in return_values[index] if "@" in i][0]
            if return_value in blacklist:
                print("Removing",return_values[index], "because in blacklist")
                del return_values[index]
            else:
                index+=1
                
    return return_values
