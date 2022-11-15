import smtplib
import yagmail
import os
import json
from tkinter import filedialog as fd
import pandas as pd
import csv

# email.json has to be in the same directory as this file
__location__= os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
def send_email(email_id : str, email_token : str, reciever :str, subject : list, body, mail_num : int = None): 
    with yagmail.SMTP(email_id, email_token) as yag:
        yag.send(reciever, subject, body)
        if mail_num:
            print('Sent email',mail_num,'successfully')
        else:
            print("Sent the email message")
def folder_selector(root):
        while True:
            filename = fd.askdirectory(title = "select a folder", initialdir = root) #,
            if filename:
                break
        return filename
data = dict(json.load(open(os.path.join(__location__,'email.json'))))
def folder_to_list(folder = None, filetype = ".csv"):
    if not folder: folder = folder_selector(__location__)
    everything = [os.path.join(dp, f) for dp, dn, fn in os.walk(folder) for f in fn]
    arr = []
    for every in everything:
        if every.endswith(filetype):
            lines=list(csv.reader(open(every)))
            lines = filter(None, lines)
            arr.extend(lines)
    return arr
def compile_and_send(reciever_data,email_data, subject, body, fillers):
    for reciever in reciever_data:
        reciever_email = reciever[-1]
        reciever_name = reciever[0]
        reciever_role = ""
        if len(reciever) > 2:
            reciever_role = ' '.join(reciever[1:-1])
            reciever_role = reciever_role.lower()
        bod = body.copy()
        for index, b in enumerate(bod):
            if b in fillers:
                bod[index] = eval(b)  
        print(email_data["user_email"],reciever_email,subject,''.join(bod), sep = "\n\n")
    
compile_and_send(
    reciever_data = folder_to_list(),
    email_data = data,
    subject  = "Online Cooking Intervention Opportunity for Students With Autism",
    body = ["Dear ", "reciever_name",",\n\nMy name is Aadvik Vashist, and I am a junior at River Hill High School here in Howard County. For the past year, I have been researching how to effectively teach individuals with Autism Daily Living Skills. From this, I have come to the conclusion that Adaptive Video Prompting, which involves using Artificial Intelligence to improve Daily Living Skill instruction, needs to be developed. I am reaching out to you, as a ", "reciever_role", ", to see if you had any students or know any people in your community that you think would be a good fit for a short, 2-3 week long program that teaches ASD individuals how to cook some simple recipes. A good fit involves: \n\n (1) A parent or guardian who is willing and able to assist with instruction.\n (2) A laptop or handheld device with Zoom installed.\n (3) Mentees who enjoy cooking and would be able to perform necessary kitchen tasks with assistance.\n\nAfter seeing how scholars research interventions but create no real-world impact, I want to be able to create substance out of my research by providing my peers the chance to learn how to cook some of their favorite dishes for free. Below I have attached a link to our website, some extra information for you, students, and parents, as well as some samples that explain the process.\n\n The link to our website is:  \n\nKind Regards,\nAadvik"],
    fillers = ["reciever_name", "reciever_role"]
)