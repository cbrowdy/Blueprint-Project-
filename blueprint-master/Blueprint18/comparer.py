import numpy as np
import cv2
import os, os.path


def getface(pictureLoc):
    cascade_file_src = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascade_file_src)
    # load image on gray scale :
    image = pictureLoc
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the image :
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    cropIm = []
    # crop face
    if len(faces) >= 1:
        x = faces[0][0]
        y = faces[0][1]
        w = faces[0][2]
        h = faces[0][3]
        for r in range(y, h + y):
            new = []
            for c in range(x, w + x):
                new.append(image[r][c])

            cropIm.append(new)
        cropIm = np.asarray(cropIm)
        cropIm = cv2.resize(cropIm, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
        cropIm = np.asarray(cropIm)
        return cropIm

def getMatch(picLoc):
    from keras.models import load_model
    model = load_model('guesser.h5')
    pic = cv2.imread(picLoc)
    pic = getface(pic)
    pred = model.predict(pic.reshape(1, 100, 100, 3))
    def getBestIndex(w):
        rating = 0
        value = 0
        for i in range(len(w)):
            if w[i] > rating:
                rating = w[i]
                value = i
        return value

    guess = getBestIndex(pred[0])
    if guess == 0:
        return ["billnye.jpeg", "Bill Nye"]
    elif guess == 1:
        return ["elonmusk.jpg", "Elon Musk"]
    elif guess == 2:
        return ["janegoodall.jpeg", "Jane Goodall"]
    elif guess == 3:
        return ["michiokaku.jpg", "Michio Kaku"]
    elif guess == 4:
        return ["neiltyson.jpg", "Neil Degrassi Tyson"]
    elif guess == 5:
        return ["sallyride.jpg", "Sally Ride"]
    elif guess == 6:
        return ["jobsGood.jpg", "Steve Jobs"]
    elif guess == 7:
        return ["swGood.jpg", "Susan Wojcicki"]
    elif guess == 8:
        return ["sagan.jpg", "Carl Sagan"]
    else:
        return ["curieGood.jpg", "Marie Curie"]