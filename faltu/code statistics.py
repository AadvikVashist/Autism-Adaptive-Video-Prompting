import os
import pathlib
from collections import Counter
import string
import re
from glob import glob
def count_letters(word, valid_letters=string.ascii_letters):
    count = Counter(word) # this counts all the letters, including invalid ones
    return sum(count[letter] for letter in valid_letters) # add up valid letters
def get_line_character_word_count(lines):
    counters = {"line_count" : 0,
                "character_count" : 0,
                "character_without_spaces_count" : 0,
                "word_count" : 0
                }
    for line in lines:
        counters["line_count"] += 1
        counters["character_count"] += len(line)
        counters["character_without_spaces_count"] += count_letters(line)
        counters["word_count"]  += len(re.split('/W+',line))
    return counters
def read_file(file):
    with open(file, "r") as filer:
        data = filer.read()
        data_into_list = data.replace('\n', ' ').split(".")
    return data_into_list
def get_all_files(root):
    file_types = [".docx", ".tex", ".readme", ".txt", ".ipynb", ".text", ".bib", ".py", ".json", ".csv", ".dat", ".db", ".dbf", ".log", ".sql", ".tar", ".xml", "jar", ".js", ".html" , ".css", ".c", ".class", ".java", ".php", ".sh", ".swift", ".h", ".cpp", ".xlsx", ".xls", ".xlsm", ".dmp", ".wpd",".rtf"]
    result = [os.path.join(dp, f).replace("\\","/") for dp, dn, filenames in os.walk(root) for f in filenames if os.path.splitext(f)[1] in file_types and "pycache" not in dp]
    counters = {"line_count" : 0,
                "character_count" : 0,
                "character_without_spaces_count" : 0,
                "word_count" : 0
                }
    for file in result:
        file_read = read_file(file)
        _ = get_line_character_word_count(file_read)
        counters["line_count"] += _["line_count"]
        counters["character_count"] += _["character_count"]
        counters["character_without_spaces_count"] += _["character_without_spaces_count"]
        counters["word_count"] += _["word_count"]
        print('/'.join(file.split("/")[-3::]), *counters.items(), "\n")
    return counters
get_all_files("C:/Users/aadvi/Desktop/Autism/Autism-Adaptive-Video-Prompting/")
    

