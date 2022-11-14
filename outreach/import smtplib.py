import smtplib
import yagmail
import os
import json
__location__= os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


data = dict(json.load(open(os.path.join(__location__,'email.json'))))

def send_email(email_id : str, email_token : str, reciever :str, subject : list, body, mail_num : int = None): 
    with yagmail.SMTP(email_id, email_token) as yag:
        yag.send(reciever, subject, body)
        if mail_num:
            print('Sent email',mail_num,'successfully')
        else:
            print("Sent the email message")
send_email(data["user_email"],data["user_password"],data["user_email"], "hola bois", "asdfsaf")

