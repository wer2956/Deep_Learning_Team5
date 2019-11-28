#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('01/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('01/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('01/haarcascade_righteye_2splits.xml')

label=['State : Drowsy', 'State : Awake']
model = load_model('a2a_final.h5')
path = os.getcwd()
font = cv2.FONT_HERSHEY_PLAIN
cap = cv2.VideoCapture(0)

a=0
b=0
c=0

count=0
score=0
thicc=2

danger = 15

rpred=[99]
lpred=[99]

start = True
    
while(True):      
    ret, frame = cap.read()
    height, width = frame.shape[:2] 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (255,100,100) , 2 )

# ------------------오른쪽 눈------------------
    for (x,y,w,h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]
        count = count+1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye, axis=0)

        rpred = model.predict_classes(r_eye)
        if(rpred[0]==1):
            label='Eyes Opened' 
        if(rpred[0]==0):
            label='Eyes Closed'
        break

# ------------------왼쪽 눈------------------
    for (x,y,w,h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]
        count = count+1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye, (24,24))
        l_eye = l_eye/255
        l_eye = l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye, axis=0)

        lpred = model.predict_classes(l_eye) # label: 0은 눈 감음, 1은 눈 뜸으로 인코딩되어 있음
        if(lpred[0]==1):
            label='Eyes Opened'   
        if(lpred[0]==0):
            label='Eyes Closed'
        break

# ------------------하단에 눈 떴음여부 & 카운트(점수) 표시 텍스트------------------        
    if(rpred[0]==0 and lpred[0]==0):
        score = score+1
        cv2.putText(frame,"State: Eyes Closed", (10, 50), font, 1.5, (255,255,255), 2, cv2.LINE_AA)
    else:
        score = score-2
        cv2.putText(frame, "State: Eyes Opened", (10, 50), font, 1.5, (255,255,255), 2, cv2.LINE_AA)

    if(score < 0):
        score = 0
    cv2.putText(frame, 'Count: ' + str(score), (10, 100), font, 1.5, (255,255,255), 2, cv2.LINE_AA)

# ------------------점수에 따른 경고창 디스플레이 조건------------------                 
    if(score < 7):
        a += 1
    
    if(score < danger and score > 7):
        cv2.putText(frame, "You're Drowsy", (20, height//2), font, 5, (255,255,255), 7, cv2.LINE_AA)
        cv2.putText(frame, 'Recommandation : Get some water', (20, height//2+100), font, 2, (255,255,255), 2, cv2.LINE_AA)
        b += 1



    if(score > danger):
        cv2.imwrite(os.path.join(path,'image.jpg'), frame)
        try:
            sound.play()
        except:  # isplaying = False
            pass
        if(thicc < 20):
            thicc = thicc+500
        else:
            thicc = thicc-500
            if(thicc < 2):
                thicc = 2
        cv2.rectangle(frame, (0,0), (width,height), (0,0,255), thicc)
        cv2.putText(frame, 'Danger', (165, height//2), font, 5, (255,255,255), 7, cv2.LINE_AA)
        cv2.putText(frame, 'Awake to be Alive', (160, height//2+100), font, 2, (255,255,255), 3, cv2.LINE_AA)
        c += 1


    cv2.imshow('frame', frame)    
    if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' : 종료
        start = False
        break

cap.release()
cv2.destroyAllWindows()


#-----------------프로그램 종료 후, 오늘의 운전상태 피드백---------------------

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rc
from matplotlib import style
while(True):
    
    group_colors = ['yellowgreen', 'lightskyblue', 'lightcoral']
    group_explodes = (0.1, 0, 0)
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
    style.use('ggplot')

    labels = ['NORMAL', 'DROWSY', 'SLEEPING']
    ratio = [a, b, c]
    fig, ax = plt.subplots()

    ax.pie(ratio, 
           explode=group_explodes,
           labels=labels, 
           colors=group_colors, 
           shadow=True, 
           startangle=90)
    ax.legend(labels, loc='upper right')
    ax.set_aspect('equal')

    plt.savefig('ex_pieplot.png', format='png', dpi=None)
    plt.show()
    
    break
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' : 종료
        start = False
        break

cv2.waitKey(0)
cv2.destroyAllWindows()


#-----------------오늘의 운전상태 피드백 후, 졸음쉼터 위치제공---------------------

from bs4 import BeautifulSoup as bs
from selenium import webdriver
import time
import numpy as np
import pandas as pd

client_id = "e9jbea0v6k"
client_secret = "2MFb9u0X1QKhDDPiClk2qCSpj5bEprzKDmNikzoo"

driver = webdriver.Chrome("chromedriver.exe")

driver.implicitly_wait(3)

area = ['경부선 졸음쉼터','서울양양선 졸음쉼터','서울외곽순환선 졸음쉼터','경상북도 졸음쉼터',
     '광주광역시 졸음쉼터','대구광역시 졸음쉼터','대전광역시 졸음쉼터','부산광역시 졸음쉼터'
        ,'서울특별시 졸음쉼터','울산광역시 졸음쉼터','인천광역시 졸음쉼터','전라남도 졸음쉼터',
        '전라북도 졸음쉼터','충청남도 졸음쉼터','충청북도 졸음쉼터','세종특별자치시 졸음쉼터']

add_name = []
for idx in area:
    driver.get('https://map.naver.com/v5/search/%EC%A1%B8%EC%9D%8C%EC%89%BC%ED%84%B0?c=14114867.9143483,4501955.5423555,14,0,0,0,dh')
    #driver.get('https://map.naver.com/v5/search/%EC%A1%B8%EC%9D%8C%EC%89%BC%ED%84%B0?c=14110913.0779308,4522082.4343128,14,0,0,0,dh')
    driver.find_element_by_css_selector('#search-input ').send_keys(idx)
    driver.fine_element_by_css_selector('#header button[type="submit"]').click()
    break


# In[ ]:




