import json
import os
import platform
from tkinter import filedialog as fd
def folder_selector(root = None):
        if not root:
            if platform.system() == 'Windows': # this hasn't been tested 
                desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop') #windows 
            elif platform.system() == 'Darwin':
                desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') #mac 
            elif platform.system() == 'Linux':
                desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') #linux
            else:
                raise Exception("Unsupported operating system: " + platform.system())
        while True:
            filename = fd.askdirectory(title = "select a folder", initialdir = desktop) #,
            if filename:
                break
        return filename
def file_selector(root = None):
        if platform.system() == 'Windows': # this hasn't been tested 
            desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop') #windows 
        elif platform.system() == 'Darwin':
            desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') #mac 
        elif platform.system() == 'Linux':
            desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') #linux
        else:
            raise Exception("Unsupported operating system: " + platform.system())
        filename = fd.askopenfilename(title = "select a file", initialdir = desktop) #,
        return filename
data_folder = folder_selector()

setting_dict = {
    "root_folder": data_folder,   
}
json_object = json.dumps(setting_dict, indent=4)
with open("settings.json", "w") as outfile:
    outfile.write(json_object)