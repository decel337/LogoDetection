from PyQt5 import QtCore

import torch
from PIL import Image
from os import listdir
import random

class DataLoader(QtCore.QThread):
    signal = QtCore.pyqtSignal('PyQt_PyObject')

    def __init__(self):
        QtCore.QThread.__init__(self)
        self.pathToTrain = ""
        self.pathToTest = ""
        self.trainData = []
        self.testData = []
        self.batchSize = 0
        self.resize = 0
        self.transforms = 0
        self.normalize = 0
        self.type = 0

    def __del__(self):
        self.wait()

    def methodSubfolders(self):
        for path in [self.pathToTrain, self.pathToTest]:
            transforms = self.transforms
            batchSize = self.batchSize
            data = []
            dataList = []
            targetList = []
            files = []
            selectedCategories = listdir(path)
            selectedCategoriesByIndex = listdir(path)
            sizeCategories = len(selectedCategories)
            numberOfFile = 0

            for directory in listdir(path):
                files.append(listdir(path + "/" + directory))
                numberOfFile += len(listdir(path + "/" + directory))

            for i in range(numberOfFile):
                dirIndex = random.randint(0, len(selectedCategoriesByIndex) - 1)
                file = random.choice(files[dirIndex])
                image = Image.open(path + "/" + selectedCategoriesByIndex[dirIndex] + "/" + file)
                image_tensor = transforms(image)
                dataList.append(image_tensor)
                target = [0] * sizeCategories
                target[selectedCategories.index(selectedCategoriesByIndex[dirIndex])] = 1
                targetList.append(target)

                if len(dataList) >= batchSize:
                    data.append((torch.stack(dataList), targetList))
                    dataList = []
                    targetList = []
                    self.signal.emit('Loaded batch ' + str(len(data)) + ' of ' + str(numberOfFile // batchSize))
                    self.signal.emit('Percentage done: ' + str(round(100 * len(data) / (numberOfFile // batchSize), 2)) + ' %')

                files[dirIndex].remove(file)
                if len(files[dirIndex]) == 0:
                    files.pop(dirIndex)
                    selectedCategoriesByIndex.pop(dirIndex)

                if path == self.pathToTrain:
                    self.trainData = data
                else:
                    self.testData = data

                if self.pathToTrain == self.pathToTest:
                    self.testData = data

                self.categories = listdir(self.pathToTrain)

    def methodByImage(self):
        for path in [self.pathToTrain, self.pathToTest]:
            batchSize = self.batchSize
            transforms = self.transforms
            data = []
            dataList = []
            targetList = []
            files = listdir(path)

            for i in range(len(listdir(path))):
                file = random.choice(files)
                files.remove(file)
                image = Image.open(path + "/" + file)
                image_tensor = transforms(image)
                dataList.append(image_tensor)
                target = []
                for category in self.categories:
                    if category in file:
                        target.append(1)
                    else:
                        target.append(0)

                targetList.append(target)

                if len(dataList) >= batchSize:
                    data.append((torch.stack(dataList), targetList))
                    dataList = []
                    targetList = []
                    self.signal.emit('Loaded batch ' + str(len(data)) + ' of ' + str(int(len(listdir(path)) / batchSize)))
                    self.signal.emit('Percentage Done: ' + str(round(100 * len(data) / int(len(listdir(path)) / batchSize), 2)) + '%')

            if path == self.pathToTrain:
                self.trainData = data
            else:
                self.testData = data

            if self.pathToTrain == self.pathToTest:
                self.testData = data

    def run(self):

        if self.type == "Picture1":
            self.methodSubfolders()
        elif self.type == "Picture2":
            self.methodByImage()
