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
            if filename is not None:
                break
        return filename
def file_selector(root = None, file_type = None):
        if root is None:
            if platform.system() == 'Windows': # this hasn't been tested 
                desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop') #windows 
            elif platform.system() == 'Darwin':
                desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') #mac 
            elif platform.system() == 'Linux':
                desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') #linux
            else:
                raise Exception("Unsupported operating system: " + platform.system())
        else:
            desktop = root
        if file_type is None:
            filename = fd.askopenfilename(title = "select a file", initialdir = desktop)
        else:
            filename = fd.askopenfilename(title = "select a file", initialdir = desktop, filetypes= file_type)

        return filename
