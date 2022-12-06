import smtplib
import yagmail
import os
import json
from tkinter import filedialog as fd
import pandas as pd
import csv
import teacher_categorizer
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
def compile_and_send_test(reciever_data,sender_data, subject, body, fillers):
    while True:
            yes_or_no = input("are you sure you want to send these emails")
            if "y" in yes_or_no:
                break
            elif "n" in yes_or_no:
                os.exit() 
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
        
        print(sender_data["user_email"],reciever_email,subject,''.join(bod), sep = "\n\n")
def compile_and_send(reciever_data,sender_data, subject, body, fillers):
    while True:
            yes_or_no = input("are you sure you want to send these emails")
            if "y" in yes_or_no:
                break
            elif "n" in yes_or_no:
                os.exit()
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
        send_email(sender_data["user_email"],sender_data["user_password"],reciever_email,subject,''.join(bod), sep = "\n\n")
def sender_data(loc = os.path.join(os.getcwd(), "settings",'email.json')):
    data = dict(json.load(open(loc)))
    return data