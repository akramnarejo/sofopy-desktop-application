from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter.ttk import Progressbar
from tkinter import messagebox
import glob
import cv2
import numpy as np
import os
import time
import threading
import imutils
import shutil


# creating main window frame named as root
root = Tk()
root.title('SOFOpy')
root.minsize(1200, 750)
root.maxsize(1200, 750)
root.resizable(0, 0)
root.configure(background='#17223b')
#root.bind('<Escape>', lambda e: root.destroy())


def on_closing():

    if messagebox.askokcancel('Quit', 'Do you want to quit'):
        root.destroy()
        shutil.rmtree(r'./Crops')
        shutil.rmtree(r'./FinalCrops')


sourcefile = ''


def chooseFile():
    print('selecting file...')
    global sourcefile
    sourcefile = filedialog.askopenfilename(
        initialdir="C:/", title="Select a File", filetypes=(("MP4 files", "*.mp4"), ("all files", "*.*")))
    #statusLbl.configure(text='file selected, click the summarize button to get analysis')
    if not sourcefile:
        statusLbl.configure(text='file is not selected.')
    else:
        statusLbl.configure(
            text='file selected, set the duration threshold for analysis.')


def clearListBox():
    listbox.delete(0, END)
    statusLbl.configure(text='Select file to summarize')
    imgViewer.configure(image="")
    imgViewer.configure(text="")
    root.update()


finalImages = []


def summarize():
    try:
        global sourcefile
        if sourcefile:
            global dur
            dur = float(enterDuration.get())
            if dur:
                statusLbl.configure(text='Summarizing...')
                root.update()
                video = cv2.VideoCapture(sourcefile)

                currentframe = 0

                while (True):
                    ret, frame = video.read()

                    if ret:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, frame.shape[0] / 1,
                                                   param1=200, param2=10, minRadius=20, maxRadius=35)
                        if circles is not None:
                            circles = np.uint16(np.around(circles))
                            for i in circles[0, :1]:
                                # draw the outer circle
                                #cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 3)
                                x = i[0] - 100
                                y = i[1] - 100
                                crop = frame[y:y + 200, x:x + 200]
                                # draw the center of the circle
                                #cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 2)
                        try:
                            # creating a folder named data
                            if not os.path.exists('Crops'):
                                os.makedirs('Crops')

                            # if not created then raise error
                        except OSError:
                            messagebox.showerror(
                                "Error", "Directory not created.")
                        check = False
                        for x in crop:
                            if x is None:
                                continue
                            else:
                                check = True

                        if check == True:
                            score = cv2.Laplacian(crop, cv2.CV_64F).var()
                            if score > 500:
                                name = './Crops/crop' + \
                                    str(currentframe) + '.jpg'
                                currentframe += 1
                                statusLbl.configure(
                                    text='Summarizing...\t' + 'Creating...' + name)
                                root.update()
                                cv2.imwrite(name, crop)

                    else:
                        break

                video.release()
                cv2.destroyAllWindows()

                merge_frame()  # calling function here

                messagebox.showinfo('message', 'Summarization completed.')
                statusLbl.configure(text='')

                global finalImages
                for file in glob.glob("./FinalCrops/*.jpg"):
                    finalImages.append(file)

                for x in finalImages:
                    listbox.insert(END, x)

                imgViewer.configure(text='click an image to \n display here.')
        else:
            messagebox.showerror('Error', 'file is not selected')

    except ValueError:
        messagebox.showerror(
            'Error', 'Invalid duration is set.\nPlease set in integers. ')
    # else:
        #messagebox.showerror('Error', 'file is not selected')


def merge_frame():
    ind = ""
    s_index = 0
    image_lists = []
    threshold = 30
    count_sim = 0
    similar_images_count = {}
    similar_list = []
    dissimliar_list = set()
    dis_index = 0
    similar_tmp = []
    similar_set = set()
    final_list_imgs = []

    for file in glob.glob("./Crops/*.jpg"):
        ind = file
        value = ind[-6]
        s_value = ind.find('p', -9, -3) + 1
        e_value = ind.find('.', -5, -2)
        s_index = (int(ind[s_value:e_value]))
        path = "./Crops/crop" + str(s_index) + ".jpg"
        image_lists.append(path)

    for i in range(len(image_lists)):
        for j in range(i + 1, len(image_lists)):

            #print(image_lists[i], image_lists[j])
            if image_lists[i] in similar_list:
                break

            elif image_lists[j] in similar_list:
                continue

            similar_tmp.append(image_lists[i])
            img = cv2.imread(image_lists[i])
            imgToCompare = cv2.imread(image_lists[j])

            #sift = cv2.xfeatures2d.SIFT_create()
            #kp1, des1 = sift.detectAndCompute(img, None)
            #kp2, des2 = sift.detectAndCompute(imgToCompare, None)

            surf = cv2.xfeatures2d.SURF_create()
            kp1, des1 = surf.detectAndCompute(img, None)
            kp2, des2 = surf.detectAndCompute(imgToCompare, None)

            index_params = dict(algorithm=0, trees=5)
            search_params = dict()
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(des1, des2, k=2)
            bestMatches = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    bestMatches.append(m)

            statusLbl.configure(
                text="Summarizing...  " + image_lists[i] + " comparing to " + image_lists[j])
            root.update()
            #print(image_lists[i], "comparing ", image_lists[j])
            dissimliar_list.add(image_lists[i])

            if len(bestMatches) > threshold:
                count_sim += 1
                similar_list.append(image_lists[j])
                similar_tmp.append(image_lists[j])
            result = cv2.drawMatches(
                img, kp1, imgToCompare, kp2, bestMatches, None)

            similar_set = set(similar_tmp)

            cmp_max = 0
            img_name = ""
            for img_ind in similar_set:
                img11 = cv2.imread(img_ind)
                # returns the amount of blurness, an image has
                score = cv2.Laplacian(img11, cv2.CV_64F).var()
                if score > cmp_max:
                    cmp_max = score
                    img_name = img_ind
            #    print("max score: ", cmp_max, img_name)
            #    final_list_imgs.append(img_name)

        similar_images_count[image_lists[i]] = count_sim
        count_sim = 0
        similar_tmp = []

    tmp = {}
    for i in dissimliar_list:
        tmp[i] = similar_images_count[i] + 1

    similar_images_count = tmp
    duration = 0.0

    try:
        # creating a folder named data
        if not os.path.exists('FinalCrops'):
            os.makedirs('FinalCrops')

        # if not created then raise error
    except OSError:
        messagebox.showerror("Error", "Directory not created.")

    # print(similar_images_count)
    for i in similar_images_count.keys():
        if len(i) > 6:
            ind = i
            value = ind[-6]
            s_value = ind.find('p', -9, -3) + 1
            e_value = ind.find('.', -5, -2)
            s_index = (int(ind[s_value:e_value]))
            d = similar_images_count[i] / 25
            if d >= dur:
                name = "./FinalCrops/" + str(d) + " secs.jpg"
                tmp_img = cv2.imread(i)
                cv2.imwrite(name, tmp_img[:])


# GUI starts

title = Label(root, text="Tobii analysis summarizer", font=(
    'arial', 28, 'bold'), fg='white', bg='#17223b')
title.place(x=280, y=20)
choosefilebtn = Button(root, text="Select File",
                       padx=10, pady=5, command=chooseFile)
choosefilebtn.place(x=80, y=90)

enterDuration = Entry(width=10, fg='black', font=('Arial', 11, 'normal'))
enterDuration.place(x=162, y=102)
enterDuration.insert(0, 'Enter here')
enterDuration.bind("<FocusIn>", lambda args: enterDuration.delete('0', 'end'))


entryLbl = Label(root, text='Set duration', fg='white', bg='#17223b')
entryLbl.place(x=162, y=80)


summarizebtn = Button(root, text="Summarize", padx=10,
                      pady=5, command=summarize)
summarizebtn.place(x=248, y=90)


statusLbl = Label(root, text='Select file to summarize', font=(
    'carier', 12, 'normal'), fg='white', bg='#17223b')
statusLbl.place(x=355, y=100)

# progressBar=Progressbar(root,length=545,orient=HORIZONTAL,maximum=100,value=50)
# progressBar.place(x=250,y=101)

listFrame = Frame(root, width=720, height=520, bg="#17223b")
listFrame.place(x=80, y=130)
imgViewFrame = Frame(root, width=320, height=530, bg="#17223b")
imgViewFrame.place(x=798, y=130)


def listItemCall():
    # item=listbox.curselection()
    global my_image
    item = listbox.get(ACTIVE)
    print(item)
    my_image = ImageTk.PhotoImage(Image.open(item))
    imgViewer.configure(image=my_image)
    # imgViewer.configure(text=item)
    #print(listbox.get(ACTIVE)," selected")


scrollbar = Scrollbar(listFrame)
scrollbar.pack(side=RIGHT, fill=Y)

listbox = Listbox(listFrame, width=116, height=33, bd=0,
                  yscrollcommand=scrollbar.set, selectmode=BROWSE)
listbox.pack(side=LEFT)
#listbox.bind("<Button-1>", listCall)
scrollbar.config(command=listbox.yview)
'''
ind=""
image_lists=[]
for file in glob.glob("./finalCrops/*.jpg"):
    ind = file
    value = ind[-6]
    s_value = ind.find('p', -9, -3) + 1
    e_value = ind.find('.', -5, -2)
    s_index = (int(ind[s_value:e_value]))
    path = "C:/Users/ma/Dropbox/FYP/sofopy/crops/crop" + str(s_index) + ".jpg"
    image_lists.append(path)
'''


def click(event):
    global pic
    itm = listbox.get(ACTIVE)
    pic = ImageTk.PhotoImage(Image.open(itm))
    imgViewer.configure(image=pic)


listbox.bind("<Button-1>", click)


imgViewer = Label(
    imgViewFrame, text='', bg='#17223b', fg='white')
imgViewer.place(x=20, y=100)

dirToSaveFiles = ''


def saveFile():
    global dirToSaveFiles
    dirToSaveFiles = filedialog.askdirectory()
    global finalImages
    for x in finalImages:
        end = x.find('secs')
        start = end - 5
        name = x[start:end]
        image = cv2.imread(x)
        cv2.imwrite(dirToSaveFiles + '/' + name + ' secs.jpg', image)


savefilebtn = Button(root, text="Save File", padx=10,
                     pady=5, command=lambda: saveFile())
savefilebtn.place(x=80, y=670)
clearbtn = Button(root, text="Clear File", padx=10,
                  pady=5, command=clearListBox)
clearbtn.place(x=160, y=670)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
