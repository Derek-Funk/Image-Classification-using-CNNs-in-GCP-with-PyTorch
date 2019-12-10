# imports & installs ---------------------------------------------------------------------------------------------------
import os
#os.system("sudo pip3 install wget") one-time statement
import zipfile
import shutil
import pandas as pd
import numpy as np
import cv2
import random

import torch
import torch.nn as nn
import tensorflow as tf
import math
from sklearn.metrics import accuracy_score

# directory set up -----------------------------------------------------------------------------------------------------
homeDirectory = '/home/ubuntu/Deep-Learning'
os.chdir(homeDirectory)
examFolder = 'finalProject'
#os.mkdir(examFolder) #one-time statement
os.chdir(examFolder)
# use os.getcwd() to get current folder
# use os.listdir() to see current folder contents

# load data ------------------------------------------------------------------------------------------------------------
# zippedFolderName = 'breast-histopathology-images.zip' #one-time statement
# zip = zipfile.ZipFile(zippedFolderName) #one-time statement
#zip.extractall() #one-time statement
#shutil.rmtree("IDC_regular_ps50_idx5") #one-time statement
#os.remove(zippedFolderName) #one-time statement
allImagesFolderName = "allImages"
#os.mkdir(allImagesFolderName) #one-time statement
# listOfPatientIds = []
# for folder in os.listdir():
#     if folder!=allImagesFolderName:
#         listOfPatientIds.append(folder)
# len(listOfPatientIds) #279 patients
# for patient in listOfPatientIds:
#     benignFolder = patient + '/0'
#     malignantFolder = patient + '/1'
#
#     listOfBenignImages = os.listdir(benignFolder)
#     for image in listOfBenignImages:
#         fileToCopy = os.path.join(benignFolder, image)
#         folderToCopyTo = os.path.join(allImagesFolderName, image)
#         shutil.copyfile(fileToCopy, folderToCopyTo)
#
#     listOfMalignantImages = os.listdir(malignantFolder)
#     for image in listOfMalignantImages:
#         fileToCopy = os.path.join(malignantFolder, image)
#         folderToCopyTo = os.path.join(allImagesFolderName, image)
#         shutil.copyfile(fileToCopy, folderToCopyTo)
#
#     shutil.rmtree(patient)
# len(os.listdir(allImagesFolderName)) #277524 images
NUMBER_OF_IMAGES = 277524
fileOfPatientIds = 'listOfPatientIds.txt'
# with open(fileOfPatientIds, 'w') as fileHandle:
#     for listItem in listOfPatientIds:
#         fileHandle.write('%s\n' % listItem)
listOfPatientIds = []
with open(fileOfPatientIds, 'r') as fileHandle:
    for line in fileHandle:
        listOfPatientIds.append(line[:-1])
listOfAllImageFilenames = os.listdir(allImagesFolderName)
lookupTable = pd.DataFrame(listOfAllImageFilenames, columns=['filename'])
lookupTable['class'] = 0
for i in np.arange(NUMBER_OF_IMAGES):
    if lookupTable['filename'][i][-5]=='1':
        lookupTable['class'][i]=1
benignLookupTable = lookupTable[lookupTable['class']==0].reset_index()
malignantLookupTable = lookupTable[lookupTable['class']==1].reset_index()
# imagePath = allImagesFolderName + '/' + listOfAllImageFilenames[17]
# x=cv2.imread(imagePath)
# cv2.imshow(winname='result', mat=imagePath)

# training, test, validation -------------------------------------------------------------------------------------------
trainingFolderName = "training"
testFolderName = "test"
validationFolderName = "validation"

# os.mkdir(trainingFolderName) #one-time statement
# os.mkdir(testFolderName) #one-time statement
# os.mkdir(validationFolderName) #one-time statement

np.random.seed(0)

# benignIndexes=np.arange(198735)
# random.shuffle(benignIndexes)
# trainingBenignIndexes = benignIndexes[:139115]
# testBenignIndexes = benignIndexes[139115:168925]
# validationBenignIndexes = benignIndexes[168925:]
#
# for image in benignLookupTable['filename'][trainingBenignIndexes]:
#     fileToCopy = os.path.join(allImagesFolderName, image)
#     folderToCopyTo = os.path.join(trainingFolderName, image)
#     shutil.copyfile(fileToCopy, folderToCopyTo)
#
# for image in benignLookupTable['filename'][testBenignIndexes]:
#     fileToCopy = os.path.join(allImagesFolderName, image)
#     folderToCopyTo = os.path.join(testFolderName, image)
#     shutil.copyfile(fileToCopy, folderToCopyTo)
#
# for image in benignLookupTable['filename'][validationBenignIndexes]:
#     fileToCopy = os.path.join(allImagesFolderName, image)
#     folderToCopyTo = os.path.join(validationFolderName, image)
#     shutil.copyfile(fileToCopy, folderToCopyTo)
#
# malignantIndexes=np.arange(78786)
# random.shuffle(malignantIndexes)
# trainingMalignantIndexes = malignantIndexes[:55151]
# testMalignantIndexes = malignantIndexes[55151:66969]
# validationMalignantIndexes = malignantIndexes[66969:]
#
# for image in malignantLookupTable['filename'][trainingMalignantIndexes]:
#     fileToCopy = os.path.join(allImagesFolderName, image)
#     folderToCopyTo = os.path.join(trainingFolderName, image)
#     shutil.copyfile(fileToCopy, folderToCopyTo)
#
# for image in malignantLookupTable['filename'][testMalignantIndexes]:
#     fileToCopy = os.path.join(allImagesFolderName, image)
#     folderToCopyTo = os.path.join(testFolderName, image)
#     shutil.copyfile(fileToCopy, folderToCopyTo)
#
# for image in malignantLookupTable['filename'][validationMalignantIndexes]:
#     fileToCopy = os.path.join(allImagesFolderName, image)
#     folderToCopyTo = os.path.join(validationFolderName, image)
#     shutil.copyfile(fileToCopy, folderToCopyTo)

# hyper parameters -----------------------------------------------------------------------------------------------------
LR = 5e-2
N_EPOCHS = 15
BATCH_SIZE = 512
DROPOUT = 0.5

# other set up ---------------------------------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# pre-processing -------------------------------------------------------------------------------------------------------
listOfTrainingImageFilenames = os.listdir(trainingFolderName)
listOfTestImageFilenames = os.listdir(testFolderName)
#######################################################################
for image in listOfTrainingImageFilenames[0:20000]:
    rawImage = cv2.imread(trainingFolderName + '/' + image)
    resizedImage = cv2.resize(rawImage, (50,50)).reshape(1,3,50,50)
    classMalignant = int(image[-5])
    if 'xTrain' not in globals():
        xTrain = resizedImage
        yTrain = classMalignant
    else:
        xTrain = np.vstack((xTrain, resizedImage))
        yTrain = np.vstack((yTrain, classMalignant))

for image in listOfTrainingImageFilenames[20000:40000]:
    rawImage = cv2.imread(trainingFolderName + '/' + image)
    resizedImage = cv2.resize(rawImage, (50,50)).reshape(1,3,50,50)
    classMalignant = int(image[-5])
    if 'xTrain2' not in globals():
        xTrain2 = resizedImage
        yTrain2 = classMalignant
    else:
        xTrain2 = np.vstack((xTrain2, resizedImage))
        yTrain2 = np.vstack((yTrain2, classMalignant))

for image in listOfTrainingImageFilenames[40000:60000]:
    rawImage = cv2.imread(trainingFolderName + '/' + image)
    resizedImage = cv2.resize(rawImage, (50,50)).reshape(1,3,50,50)
    classMalignant = int(image[-5])
    if 'xTrain3' not in globals():
        xTrain3 = resizedImage
        yTrain3 = classMalignant
    else:
        xTrain3 = np.vstack((xTrain3, resizedImage))
        yTrain3 = np.vstack((yTrain3, classMalignant))

for image in listOfTrainingImageFilenames[60000:80000]:
    rawImage = cv2.imread(trainingFolderName + '/' + image)
    resizedImage = cv2.resize(rawImage, (50,50)).reshape(1,3,50,50)
    classMalignant = int(image[-5])
    if 'xTrain4' not in globals():
        xTrain4 = resizedImage
        yTrain4 = classMalignant
    else:
        xTrain4 = np.vstack((xTrain4, resizedImage))
        yTrain4 = np.vstack((yTrain4, classMalignant))

for image in listOfTrainingImageFilenames[80000:100000]:
    rawImage = cv2.imread(trainingFolderName + '/' + image)
    resizedImage = cv2.resize(rawImage, (50,50)).reshape(1,3,50,50)
    classMalignant = int(image[-5])
    if 'xTrain5' not in globals():
        xTrain5 = resizedImage
        yTrain5 = classMalignant
    else:
        xTrain5 = np.vstack((xTrain5, resizedImage))
        yTrain5 = np.vstack((yTrain5, classMalignant))

for image in listOfTrainingImageFilenames[100000:120000]:
    rawImage = cv2.imread(trainingFolderName + '/' + image)
    resizedImage = cv2.resize(rawImage, (50,50)).reshape(1,3,50,50)
    classMalignant = int(image[-5])
    if 'xTrain6' not in globals():
        xTrain6 = resizedImage
        yTrain6 = classMalignant
    else:
        xTrain6 = np.vstack((xTrain6, resizedImage))
        yTrain6 = np.vstack((yTrain6, classMalignant))

for image in listOfTrainingImageFilenames[120000:140000]:
    rawImage = cv2.imread(trainingFolderName + '/' + image)
    resizedImage = cv2.resize(rawImage, (50,50)).reshape(1,3,50,50)
    classMalignant = int(image[-5])
    if 'xTrain7' not in globals():
        xTrain7 = resizedImage
        yTrain7 = classMalignant
    else:
        xTrain7 = np.vstack((xTrain7, resizedImage))
        yTrain7 = np.vstack((yTrain7, classMalignant))

for image in listOfTrainingImageFilenames[140000:160000]:
    rawImage = cv2.imread(trainingFolderName + '/' + image)
    resizedImage = cv2.resize(rawImage, (50,50)).reshape(1,3,50,50)
    classMalignant = int(image[-5])
    if 'xTrain8' not in globals():
        xTrain8 = resizedImage
        yTrain8 = classMalignant
    else:
        xTrain8 = np.vstack((xTrain8, resizedImage))
        yTrain8 = np.vstack((yTrain8, classMalignant))

for image in listOfTrainingImageFilenames[160000:180000]:
    rawImage = cv2.imread(trainingFolderName + '/' + image)
    resizedImage = cv2.resize(rawImage, (50,50)).reshape(1,3,50,50)
    classMalignant = int(image[-5])
    if 'xTrain9' not in globals():
        xTrain9 = resizedImage
        yTrain9 = classMalignant
    else:
        xTrain9 = np.vstack((xTrain9, resizedImage))
        yTrain9 = np.vstack((yTrain9, classMalignant))

for image in listOfTrainingImageFilenames[180000:]:
    rawImage = cv2.imread(trainingFolderName + '/' + image)
    resizedImage = cv2.resize(rawImage, (50,50)).reshape(1,3,50,50)
    classMalignant = int(image[-5])
    if 'xTrain10' not in globals():
        xTrain10 = resizedImage
        yTrain10 = classMalignant
    else:
        xTrain10 = np.vstack((xTrain10, resizedImage))
        yTrain10 = np.vstack((yTrain10, classMalignant))

##############

for image in listOfTestImageFilenames[0:2000]:
    rawImage = cv2.imread(testFolderName + '/' + image)
    resizedImage = cv2.resize(rawImage, (50,50)).reshape(1,3,50,50)
    classMalignant = int(image[-5])
    if 'xTest' not in globals():
        xTest = resizedImage
        yTest = classMalignant
    else:
        xTest = np.vstack((xTest, resizedImage))
        yTest = np.vstack((yTest, classMalignant))

# CNN model ------------------------------------------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (2,2))  # output (n_examples, 16, 49, 49)
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 2))  # output (n_examples, 16, 24, 24)
        self.conv2 = nn.Conv2d(16, 32, (2,2))  # output (n_examples, 32, 23, 23)
        self.convnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((2, 2))  # output (n_examples, 32, 11, 11)
        self.linear1 = nn.Linear(32*11*11, 400) # output (n_examples, 400)
        self.linear1_bn = nn.BatchNorm1d(400)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(400, 2)
        self.act = nn.LogSoftmax(1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.convnorm1(x)
        x = self.pool1(x)
        x = self.act(self.conv2(x))
        x = self.convnorm2(x)
        x = self.pool2(x)
        x = self.act(self.linear1(x.view(len(x), -1)))
        x = self.linear1_bn(x)
        x = self.drop(x)
        x = self.linear2(x)
        return x

finalModel = CNN().to(device)
optimizer = torch.optim.SGD(finalModel.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# %% ----------------------------------- Helper Functions --------------------------------------------------------------
def acc(x, y, return_labels=False):
    with torch.no_grad():
        logits = finalModel(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y.cpu().numpy(), pred_labels)

# training segment -----------------------------------------------------------------------------------------------------
print("Starting training loop...")
for epoch in range(N_EPOCHS):
    loss_train = 0
    finalModel.train()

    xTrainTensor = torch.tensor(xTrain).float().to(device)
    yTrainTensor = torch.tensor(yTrain).float().to(device)
    for batch in range(len(xTrainTensor)//BATCH_SIZE):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = finalModel(xTrainTensor[inds])
        target = yTrainTensor[inds].long().squeeze()
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    del xTrainTensor, yTrainTensor

    xTrainTensor2 = torch.tensor(xTrain2).float().to(device)
    yTrainTensor2 = torch.tensor(yTrain2).float().to(device)
    for batch in range(len(xTrainTensor2)//BATCH_SIZE):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = finalModel(xTrainTensor2[inds])
        target = yTrainTensor2[inds].long().squeeze()
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    del xTrainTensor2, yTrainTensor2

    xTrainTensor3 = torch.tensor(xTrain3).float().to(device)
    yTrainTensor3 = torch.tensor(yTrain3).float().to(device)
    for batch in range(len(xTrainTensor3) // BATCH_SIZE):
        inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
        optimizer.zero_grad()
        logits = finalModel(xTrainTensor3[inds])
        target = yTrainTensor3[inds].long().squeeze()
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    del xTrainTensor3, yTrainTensor3

    xTrainTensor4 = torch.tensor(xTrain4).float().to(device)
    yTrainTensor4 = torch.tensor(yTrain4).float().to(device)
    for batch in range(len(xTrainTensor4)//BATCH_SIZE):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = finalModel(xTrainTensor4[inds])
        target = yTrainTensor4[inds].long().squeeze()
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    del xTrainTensor4, yTrainTensor4

    xTrainTensor5 = torch.tensor(xTrain5).float().to(device)
    yTrainTensor5 = torch.tensor(yTrain5).float().to(device)
    for batch in range(len(xTrainTensor5)//BATCH_SIZE):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = finalModel(xTrainTensor5[inds])
        target = yTrainTensor5[inds].long().squeeze()
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    del xTrainTensor5, yTrainTensor5

    xTrainTensor6 = torch.tensor(xTrain6).float().to(device)
    yTrainTensor6 = torch.tensor(yTrain6).float().to(device)
    for batch in range(len(xTrainTensor6)//BATCH_SIZE):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = finalModel(xTrainTensor6[inds])
        target = yTrainTensor6[inds].long().squeeze()
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    del xTrainTensor6, yTrainTensor6

    xTrainTensor7 = torch.tensor(xTrain7).float().to(device)
    yTrainTensor7 = torch.tensor(yTrain7).float().to(device)
    for batch in range(len(xTrainTensor7)//BATCH_SIZE):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = finalModel(xTrainTensor7[inds])
        target = yTrainTensor7[inds].long().squeeze()
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    del xTrainTensor7, yTrainTensor7

    xTrainTensor8 = torch.tensor(xTrain8).float().to(device)
    yTrainTensor8 = torch.tensor(yTrain8).float().to(device)
    for batch in range(len(xTrainTensor8)//BATCH_SIZE):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = finalModel(xTrainTensor8[inds])
        target = yTrainTensor8[inds].long().squeeze()
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    del xTrainTensor8, yTrainTensor8

    xTrainTensor9 = torch.tensor(xTrain9).float().to(device)
    yTrainTensor9 = torch.tensor(yTrain9).float().to(device)
    for batch in range(len(xTrainTensor9) // BATCH_SIZE):
        inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
        optimizer.zero_grad()
        logits = finalModel(xTrainTensor9[inds])
        target = yTrainTensor9[inds].long().squeeze()
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    del xTrainTensor9, yTrainTensor9

    xTrainTensor10 = torch.tensor(xTrain10).float().to(device)
    yTrainTensor10 = torch.tensor(yTrain10).float().to(device)
    for batch in range(len(xTrainTensor10) // BATCH_SIZE):
        inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
        optimizer.zero_grad()
        logits = finalModel(xTrainTensor10[inds])
        target = yTrainTensor10[inds].long().squeeze()
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    del xTrainTensor10, yTrainTensor10

    xTestTensor = torch.tensor(xTest).float().to(device)
    yTestTensor = torch.tensor(yTest).float().to(device)
    finalModel.eval()
    with torch.no_grad():
        y_test_pred = finalModel(xTestTensor)
        loss = criterion(y_test_pred, yTestTensor.long().squeeze())
        loss_test = loss.item()

    print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
        epoch, loss_train / BATCH_SIZE, 0, loss_test, acc(xTestTensor, yTestTensor)))
    del xTestTensor, yTestTensor

# save model -----------------------------------------------------------------------------------------------------------
modelFilename = 'model_derek_funk_4.pt'
torch.save(finalModel.state_dict(), modelFilename)