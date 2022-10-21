import random
import cv2
import mediapipe as mp
import pandas as pd  
import os
import numpy as np
import time
import cvzone

def image_processed(hand_img):

    # Image processing
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # accessing MediaPipe solutions
    mp_hands = mp.solutions.hands

    # Initialize Hands
    hands = mp_hands.Hands(static_image_mode=True,
    max_num_hands=1, min_detection_confidence=0.7)
    

    # Results
    output = hands.process(img_rgb)

    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        #print(data)
        data = str(data)

        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)
                        
        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return(clean)
    except:
        return(np.zeros([1,63], dtype=int)[0])

import pickle
# load model
with open('model.pkl', 'rb') as file:
    svm = pickle.load(file)


import cv2 as cv
cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
if not cap.isOpened():
    print("Cannot open camera")
    exit()   

# mendeklarasikan variable variable kontrol seperti timer,state result, startgame dan list score
timer = 0
stateResult = False
startGame = False
scores = [0, 0]  # [AI, Player]

# looping untuk menjalankan program game rock paper scissor
while True:
    # mengatur pembacaan kamera dan background aplikasi
    imgBG = cv2.imread("Resources/BG.png")
    success, img = cap.read()

    imgScaled = cv2.resize(img, (0, 0), None, 0.875, 0.875)
    imgScaled = imgScaled[:, 80:480]

    data = image_processed(imgScaled)
    
    data = np.array(data)
    y_pred = svm.predict(data.reshape(-1,63))
    

    if startGame:
        cv2.putText(imgScaled, str(y_pred[0]),(30,40), cv2.FONT_HERSHEY_PLAIN, 
                    2, (255, 0, 0), 2, cv2.LINE_AA)

        if stateResult is False:
            timer = time.time() - initialTime
            cv2.putText(imgBG, str(int(timer)), (605, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)

            if timer > 3:
                stateResult = True
                timer = 0

                playerMove = None
                if y_pred == 'Rock':
                    playerMove = 1
                if y_pred == 'Paper':
                    playerMove = 2
                if y_pred == 'Scissor':
                    playerMove = 3

                randomNumber = random.randint(1, 3)
                imgAI = cv2.imread(f'Resources/{randomNumber}.png', cv2.IMREAD_UNCHANGED)
                imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

                # Player Wins
                if (playerMove == 1 and randomNumber == 3) or \
                        (playerMove == 2 and randomNumber == 1) or \
                        (playerMove == 3 and randomNumber == 2):
                    scores[1] += 1

                # AI Wins
                if (playerMove == 3 and randomNumber == 1) or \
                        (playerMove == 1 and randomNumber == 2) or \
                        (playerMove == 2 and randomNumber == 3):
                    scores[0] += 1
                

    imgBG[234:654, 795:1195] = imgScaled

    if stateResult:
        imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

    cv2.putText(imgBG, str(scores[0]), (410, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
    cv2.putText(imgBG, str(scores[1]), (1112, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)

    # cv2.imshow("Image", img)
    cv2.imshow("BG", imgBG)
    # cv2.imshow("Scaled", imgScaled)

    key = cv2.waitKey(1)
    if key == ord('s'):
        startGame = True
        initialTime = time.time()
        stateResult = False
    elif key == ord('q'):
        cap.release()
