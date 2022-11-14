import pandas as pd
import numpy as np
dict_schools = {}
with open('/Users/aadvik/Desktop/schools_list.txt', newline= "\n") as f:
    value = []
    for line in f.readlines():
        if "|" not in line:
            try:
                dict_schools[key] = value
            except:
                key = ""
                pass
            if '\n' in line:
                line = line[0:-1]
            key = line
            
            value = []
        else:
            row = []
            line = line.split("|")
            email = line[-1]
            if '\n' in email:
                email = email[0:-1]
            row.append(line[0].strip())
            row.append(email)
            if len(line) == 3:
                row.append(line[1].strip())
            elif len(line) > 3:
                lengthy = line[1:-1]
                for i in lengthy:
                    row.append(i.strip())
            value.append(row)
asdj = 0