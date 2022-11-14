import smtplib
import yagmail
import os
import json
from tkinter import filedialog as fd
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

def compile_and_send(email_data):
    subject  = "Online Cooking Intervention Opportunity for Students With Autism"
    individual_name = "Sierra Collis"
    job = "Philanthropist"
    body = f"Dear {individual_name}, \n\n My name is Aadvik Vashist, and I am a junior at River Hill High School here in Howard County. For the past year, I have been researching how to effectively teach individuals with Autism Daily Living Skills. From this, I have come to the conclusion that Adaptive Video Prompting, which involves using Artificial Intelligence to improve Daily Living Skill instruction, needs to be developed. I am reaching out to you, as a {job}, to see if you had any students or know any people in your community that you think would be a good fit for a short, 2-3 week long program that teaches ASD individuals how to cook some simple recipes. A good fit involves: \n (1) A parent or guardian who is willing and able to assist with instruction.\n (2) A laptop or handheld device with Zoom installed.\n (3) Mentees who enjoy cooking and would be able to perform necessary kitchen tasks with assistance.\n After seeing how scholars research interventions but create no real-world impact, I want to be able to create substance out of my research by providing my peers the chance to learn how to cook some of their favorite dishes for free. Below I have attached a link to our website, some extra information for you, students, and parents, as well as some samples that explain the process.\n\n The link to our website is:  \n\nKind Regards,\nAadvik"
    reciever_email = "scollis.06@gmail.com"
    send_email(email_data["user_email"],email_data["user_password"],reciever_email,subject, body)

compile_and_send(data)