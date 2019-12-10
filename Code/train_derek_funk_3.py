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
N_EPOCHS = 30
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

for image in listOfTestImageFilenames[0:5000]:
    rawImage = cv2.imread(testFolderName + '/' + image)
    resizedImage = cv2.resize(rawImage, (50,50)).reshape(1,3,50,50)
    classMalignant = int(image[-5])
    if 'xTest' not in globals():
        xTest = resizedImage
        yTest = classMalignant
    else:
        xTest = np.vstack((xTest, resizedImage))
        yTest = np.vstack((yTest, classMalignant))

# del xTrain,xTest,yTrain,yTest

# save as tensors to device --------------------------------------------------------------------------------------------
xTrainTensor = torch.tensor(xTrain).float().to(device)
xTestTensor = torch.tensor(xTest).float().to(device)
yTrainTensor = torch.tensor(yTrain).float().to(device)
yTestTensor = torch.tensor(yTest).float().to(device)
#xTrainTensor.requires_grad = True

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
        # self.act = torch.relu
        self.act = nn.LogSoftmax(1)

    def forward(self, x):
        # x = self.act(self.conv1(x))
        x = self.act(self.conv1(x))
        x = self.convnorm1(x)
        x = self.pool1(x)
        # x = self.act(self.conv2(x))
        x = self.act(self.conv2(x))
        x = self.convnorm2(x)
        x = self.pool2(x)
        # x = self.act(self.linear1(x.view(len(x), -1)))
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
    for batch in range(len(xTrainTensor)//BATCH_SIZE):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = finalModel(xTrainTensor[inds])
        target = yTrainTensor[inds].long().squeeze()
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    finalModel.eval()
    with torch.no_grad():
        y_test_pred = finalModel(xTestTensor)
        loss = criterion(y_test_pred, yTestTensor.long().squeeze())
        loss_test = loss.item()

    # print('Epoch ' + str(epoch) + ' complete')
    print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
        epoch, loss_train / BATCH_SIZE, 0, loss_test, acc(xTestTensor, yTestTensor)))

# save model -----------------------------------------------------------------------------------------------------------
modelFilename = 'model_derek_funk_3.pt'
torch.save(finalModel.state_dict(), modelFilename)