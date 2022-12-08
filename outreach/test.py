import teacher_categorizer
import send_emails
import re

reciever_data, jobs, blacklist = teacher_categorizer.folder_to_list() 
sender_data = send_emails.sender_data()
a = teacher_categorizer.categorization(reciever_data, jobs[1], match_percentage = 0.7,  blacklist= blacklist)
send_emails.compile_and_send_test(
    reciever_data = a,
    sender_data = sender_data,
    subject  = "Online Cooking Intervention Opportunity for Students With Autism",
    body = ["Dear ", "reciever_name",",\n\nMy name is Aadvik Vashist, and I am a junior at River Hill High School here in Howard County. For the past year, I have been researching how to effectively teach individuals with Autism Daily Living Skills. From this, I have come to the conclusion that Adaptive Video Prompting, which involves using Artificial Intelligence to improve Daily Living Skill instruction, needs to be developed. I am reaching out to you, as a[n]", "reciever_role", ", to see if you had any students or know any people in your community that you think would be a good fit for a short, 2-3 week long program that teaches ASD individuals how to cook some simple recipes. A good fit involves: \n\n (1) A parent or guardian who is willing and able to assist with instruction.\n (2) A laptop or handheld device with Zoom installed.\n (3) Mentees who enjoy cooking and would be able to perform necessary kitchen tasks with assistance.\n\nAfter seeing how scholars research interventions but create no real-world impact, I want to be able to create substance out of my research by providing my peers the chance to learn how to cook some of their favorite dishes for free. Below I have attached a link to our website, some extra information for you, students, and parents, as well as some samples that explain the process.\n\n The link to our website is:  \n\nKind Regards,\nAadvik"],
    fillers = ["reciever_name", "reciever_role"]
)