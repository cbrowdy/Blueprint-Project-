import numpy as np
import cv2
import os, os.path
from keras import Input
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, K
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop

cascade_file_src = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascade_file_src)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y, d):
    """ Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    return K.mean(y * K.square(d) + (1 - y) * K.square(K.maximum(margin - d, 0)))

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
training_expected = np.empty([len(training_inputs), 100, 100, 3])
training_inputs = np.concatenate((training_inputs, np.empty([len(training_inputs), 100, 100, 3])))
for count in range(bn):
    training_expected[count] = training_inputs[0]
    training_inputs[len(training_inputs)-count-1] = training_inputs[count]
for count in range(bn, em):
    training_expected[count] = training_inputs[bn]
    training_inputs[len(training_inputs) - count - 1] = training_inputs[count]
for count in range(em,jg):
    training_expected[count] = training_inputs[em]
    training_inputs[len(training_inputs) - count - 1] = training_inputs[count]
for count in range(jg, mk):
    training_expected[count] = training_inputs[jg]
    training_inputs[len(training_inputs) - count - 1] = training_inputs[count]
for count in range(mk, nt):
    training_expected[count] = training_inputs[mk]
    training_inputs[len(training_inputs) - count - 1] = training_inputs[count]
for count in range(nt, sr):
    training_expected[count] = training_inputs[nt]
    training_inputs[len(training_inputs) - count - 1] = training_inputs[count]
for count in range(sr, sj):
    training_expected[count] = training_inputs[sr]
    training_inputs[len(training_inputs) - count - 1] = training_inputs[count]
for count in range(sj, sw):
    training_expected[count] = training_inputs[sj]
    training_inputs[len(training_inputs) - count - 1] = training_inputs[count]
for count in range(sw, cs):
    training_expected[count] = training_inputs[sw]
    training_inputs[len(training_inputs) - count - 1] = training_inputs[count]
for count in range(cs, mc):
    training_expected[count] = training_inputs[cs]
    training_inputs[len(training_inputs) - count - 1] = training_inputs[count]

output = np.concatenate((np.zeros(round(len(training_inputs)/2)), np.ones(round(len(training_inputs)/2))))
training_expected = np.concatenate((training_expected, training_expected))

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
# model.add(Dense(256, activation='softmax'))
#
#
# input_a = Input(shape=(100, 100, 3))
# input_b = Input(shape=(100, 100, 3))
# processed_a = model(input_a)
# processed_b = model(input_b)
#
# distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
#
# siamese = Model(input=[input_a, input_b], output=distance)
# rms = RMSprop()
# siamese.compile(loss=contrastive_loss, optimizer=rms)
#
# siamese.fit([training_inputs, training_expected], output, batch_size=50, epochs=4, verbose=1)

import h5py
# model.save("guessersiam.h5")
from keras.models import load_model
model = load_model('guessersiam.h5')
def getBestIndex(w):
    predbn = model.predict(training_inputs[0].reshape(1, 100, 100, 3), training_expected[bn].reshape(1, 100, 100, 3))
    predem = model.predict(training_inputs[0].reshape(1, 100, 100, 3), training_expected[em].reshape(1, 100, 100, 3))
    predjg = model.predict(training_inputs[0].reshape(1, 100, 100, 3), training_expected[jg].reshape(1, 100, 100, 3))
    predmk = model.predict(training_inputs[0].reshape(1, 100, 100, 3), training_expected[mk].reshape(1, 100, 100, 3))
    prednt = model.predict(training_inputs[0].reshape(1, 100, 100, 3), training_expected[nt].reshape(1, 100, 100, 3))
    predsr = model.predict(training_inputs[0].reshape(1, 100, 100, 3), training_expected[sr].reshape(1, 100, 100, 3))
    predsj = model.predict(training_inputs[0].reshape(1, 100, 100, 3), training_expected[sj].reshape(1, 100, 100, 3))
    predsw = model.predict(training_inputs[0].reshape(1, 100, 100, 3), training_expected[sw].reshape(1, 100, 100, 3))
    predcs = model.predict(training_inputs[0].reshape(1, 100, 100, 3), training_expected[cs].reshape(1, 100, 100, 3))
    predmc = model.predict(training_inputs[0].reshape(1, 100, 100, 3), training_expected[mc].reshape(1, 100, 100, 3))
    preds = [predbn, predem, predjg, predmk, prednt, predsr, predsj, predsw, predcs, predmc]
    bestPred = preds[0][0]
    bestPredIndex = 0
    for i in range(1, len(preds)):
        if preds[i][0] < bestPred:
            bestPred = preds[i][0]
            bestPredIndex = i
    return bestPredIndex


guess = getBestIndex(training_inputs[3])
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

score = siamese.evaluate(training_inputs, training_expected, verbose=0)
print(score[1])
