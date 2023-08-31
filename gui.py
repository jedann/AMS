from tkinter import * 
import os
from face_extract import face_detect
from face_recognization import face_recognize

root = Tk()
root.geometry('480x360')
root.resizable(width=True,height=True)


def recognize():
    r = face_recognize()
    r.face()

def detect():
    label = name_entered.get()
    print(label)
    r = face_detect
    r.face_detected(label)
    

def click():
    label = name_entered.get()
    if not os.path.exists(label):
        os.makedirs('./database/'+label)
        

capture = Button(root,text='CAPTURE',command=detect)

Label(root,text='NAME').pack()
name = StringVar()
name_entered = Entry(root,width=20,textvariable=name)
print(name_entered)
name_entered.pack()
name_entered.focus()



Label(root,text='ROLL NO').pack()
roll_no = StringVar()
roll_no_entered = StringVar()
roll_no_entered = Entry(root,width=10,textvariable=roll_no)
print(roll_no_entered)
roll_no_entered.pack()

Label(root,text=' ').pack()

enter = Button(root,text='SAVE_DIRECTORY',command=click)
enter.pack()

Label(root, text=' ').pack()

capture.pack()
Label(root, text=' ').pack()

reco = Button(root,text='ADD PRESENT',command=recognize)
reco.pack()

root.mainloop()
