import numpy as np
from matplotlib import pyplot as plt
import cv2
import os, os.path
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
cascade_file_src = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascade_file_src)


def getface(imagepath):
    # load image on gray scale :
    image = cv2.imread(imagepath)
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


def getAllImages(impath):
    imageDir = impath
    image_path_list = []
    valid_image_extensions = [".jpeg"]  # specify your vald extensions here
    valid_image_extensions = [item.lower() for item in valid_image_extensions]
    for file in os.listdir(imageDir):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue
        image_path_list.append(os.path.join(imageDir, file))
    images = np.empty([len(image_path_list), 100, 100, 3])
    counter = 0
    for imagePath in image_path_list:
        img = getface(imagePath)
        if img is None:
            continue
        images[counter] = img
        counter += 1
    images = images[0:counter]
    return images

training_inputs = getAllImages("billnye")
bn = len(training_inputs)
training_inputs = np.concatenate((training_inputs, getAllImages("elonmusk")))
em = len(training_inputs)
training_inputs = np.concatenate((training_inputs, getAllImages("janegoodall")))
jg = len(training_inputs)
training_inputs = np.concatenate((training_inputs, getAllImages("michiokaku")))
mk = len(training_inputs)
training_inputs = np.concatenate((training_inputs, getAllImages("neiltyson")))
nt = len(training_inputs)
training_inputs = np.concatenate((training_inputs, getAllImages("sallyride")))
sr = len(training_inputs)
training_inputs = np.concatenate((training_inputs, getAllImages("steveJobs")))
sj = len(training_inputs)
training_inputs = np.concatenate((training_inputs, getAllImages("susanWoj")))
sw = len(training_inputs)
training_inputs = np.concatenate((training_inputs, getAllImages("carlSagan")))
cs = len(training_inputs)
training_inputs = np.concatenate((training_inputs, getAllImages("marieCurie")))
mc = len(training_inputs)

training_expected = np.empty([len(training_inputs), 10])

for count in range(bn):
    training_expected[count] = np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

for count in range(bn, em):
    training_expected[count] = np.asarray([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

for count in range(em,jg):
    training_expected[count] = np.asarray([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

for count in range(jg, mk):
    training_expected[count] = np.asarray([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

for count in range(mk, nt):
    training_expected[count] = np.asarray([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

for count in range(nt, sr):
    training_expected[count] = np.asarray([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

for count in range(sr, sj):
    training_expected[count] = np.asarray([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])

for count in range(sj, sw):
    training_expected[count] = np.asarray([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

for count in range(sw, cs):
    training_expected[count] = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])

for count in range(cs, mc):
    training_expected[count] = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

#network
# model = Sequential()
#
# model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(100, 100, 3)))
# model.add(Convolution2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Convolution2D(128, (3, 3), activation='relu', input_shape=(100, 100, 3)))
# model.add(Convolution2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Convolution2D(256, (3, 3), activation='relu', input_shape=(100, 100, 3)))
# model.add(Convolution2D(256, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

from keras.models import load_model
model = load_model('guesser.h5')

# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(training_inputs, training_expected,
          batch_size=150, epochs=10, verbose=1)
import h5py
model.save("guesser.h5")
pred = model.predict(training_inputs[2].reshape(1, 100, 100, 3))
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
    print("Bill Nye")
elif guess ==1:
    print("Elon Musk")
elif guess == 2:
    print("Jane Goodall")
elif guess == 3:
    print("Michu Kaku")
elif guess == 4:
    print("Neil Tyson")
elif guess == 5:
    print("Sally Ride")
elif guess == 6:
    print("Steve Jobs")
elif guess == 7:
    print("Susan Wojcicki")
elif guess == 8:
    print("Carl Sagan")
else:
    print("Marie Curie")

score = model.evaluate(training_inputs, training_expected, verbose=0)
print(score[1])
