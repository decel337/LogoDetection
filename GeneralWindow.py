from PyQt5 import QtCore, QtGui, QtWidgets
from WorldWidget import *
from LayerMenu import *
from DataLoadCfgDialog import *
from DataLoader import *
from Layer import *
from LayerDialog import *
from GeneralWindow import *
from NetworkTrainer import *
from Network import *
from TestPictureDialog import *
from WorldWidget import *

import torch
import torchvision
from torchvision import transforms
from PIL import Image
from os import listdir
import random
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
import time
import os
import sys


class GeneralWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.networkBuilder = NetworkBuilder([], [], -1)
        self.networkBuilder.AddLayer("Linear Layer")
        self.setMouseTracking(True)
        self.mousePosition = 0
        self.isResize = False
        self.dialogBox = "Dialog object"
        self.dataThread = DataLoader()
        self.networkThread = NetworkTrainer()
        self.initUserInterface()

    def initUserInterface(self):
        self.setObjectName("GeneralWindow")
        self.resize(899, 799)

        #config horizontal
        self.generalWindowHorizontal = QtWidgets.QWidget(self)
        self.generalWindowHorizontal.setObjectName("generalWindowHorizontal")

        #config vertical
        self.generalWindowVertical = QtWidgets.QVBoxLayout(self.generalWindowHorizontal)
        self.generalWindowVertical.setObjectName("generalWindowVertical")
        self.generalWindowVertical.setSpacing(30)
        self.generalWindowVertical.setContentsMargins(30, 30, 30, 30)


        self.scrollNetworkBuilder = QtWidgets.QScrollArea(self.generalWindowHorizontal)
        self.scrollNetworkBuilder.setObjectName("scrollNetworkBuilder")
        self.scrollNetworkBuilder.setStyleSheet("background-color: rgb(255, 255, 255);")

        self.scrollNetworkBuilder.setWidget(self.networkBuilder)
        self.generalWindowVertical.addWidget(self.scrollNetworkBuilder)

        self.scrollLog = QtWidgets.QScrollArea(self.generalWindowHorizontal)
        self.scrollLog.verticalScrollBar().rangeChanged.connect(self.scrollToBottom, )
        self.scrollLog.setObjectName("scrollLog")
        self.scrollLog.setMaximumSize(QtCore.QSize(16767235, 100))
        self.scrollLog.setStyleSheet("background-color: rgb(255, 255, 255);")


        self.logger = Logger([])
        self.initLogger()

        self.scrollLog.setWidget(self.logger)
        self.generalWindowVertical.addWidget(self.scrollLog)
        #change
        self.buttonClearLog = QtWidgets.QPushButton(self.generalWindowHorizontal)
        self.buttonClearLog.setMaximumSize(100, 50)
        self.buttonClearLog.setText("Clear")
        self.buttonClearLog.clicked.connect(self.clearLog)

        self.generalWindowVertical.addWidget(self.buttonClearLog)

        self.generalWindowHorizontal.setMouseTracking(True)
        self.setCentralWidget(self.generalWindowHorizontal)

        self.menuItems = QtWidgets.QMenuBar(self)
        self.menuItems.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.menuItems.setGeometry(QtCore.QRect(0, 0, 931, 24))

        self.menuFile = QtWidgets.QMenu(self.menuItems)
        self.menuFile.setTitle("File")

        self.menuLayer = QtWidgets.QMenu(self.menuItems)
        self.menuLayer.setTitle("Layers")

        self.menuFunctions = QtWidgets.QMenu(self.menuItems)
        self.menuFunctions.setTitle("Functions")

        self.menuData = QtWidgets.QMenu(self.menuItems)
        self.menuData.setTitle("Data")

        self.menuTest = QtWidgets.QMenu(self.menuItems)
        self.menuTest.setTitle("Test")

        self.setMenuBar(self.menuItems)

        self.statusForm = QtWidgets.QStatusBar(self)
        self.statusForm.setObjectName("statusForm")
        self.setStatusBar(self.statusForm)

        self.loadNetworkAction = QtWidgets.QAction(self)
        self.loadNetworkAction.setText("Load Network")
        self.loadNetworkAction.triggered.connect(self.loadNetwork)

        self.saveNetworkAction = QtWidgets.QAction(self)
        self.saveNetworkAction.setText("Save Network")
        self.saveNetworkAction.triggered.connect(self.saveNetworkBeforeTrain)

        self.ExportPyAction = QtWidgets.QAction(self)
        self.ExportPyAction.setText("Export .py")
        self.ExportPyAction.triggered.connect(self.ExportNetworkPy)

        self.convAction = QtWidgets.QAction(self)
        self.convAction.setText("Convolution Layer")
        self.convAction.triggered.connect(lambda: self.buildLayer("Convolution Layer"))

        self.linearAction = QtWidgets.QAction(self)
        self.linearAction.setText("Linear Layer")
        self.linearAction.triggered.connect(lambda: self.buildLayer("Linear Layer"))

        self.reluAction = QtWidgets.QAction(self)
        self.reluAction.setText(("ReLu"))
        self.reluAction.triggered.connect(lambda: self.buildLayer("ReLu"))

        self.sigmoidAction = QtWidgets.QAction(self)
        self.sigmoidAction.setText("Sigmoid")
        self.sigmoidAction.triggered.connect(lambda: self.buildLayer("Sigmoid"))

        self.maxPoolAction = QtWidgets.QAction(self)
        self.maxPoolAction.setText("Max Pool")
        self.maxPoolAction.triggered.connect(lambda: self.buildLayer("Max Pool"))

        self.avgPoolAction = QtWidgets.QAction(self)
        self.avgPoolAction.setText("Avg Pool")
        self.avgPoolAction.triggered.connect(lambda: self.buildLayer("Avg Pool"))

        self.viewAction = QtWidgets.QAction(self)
        self.viewAction.setText("View")
        self.viewAction.triggered.connect(lambda: self.buildLayer("View"))

        self.dropoutAction = QtWidgets.QAction(self)
        self.dropoutAction.setText("Dropout")
        self.dropoutAction.triggered.connect(lambda: self.buildLayer("Dropout"))

        self.picture1Action = QtWidgets.QAction(self)
        self.picture1Action.setText("Picture1")
        self.picture1Action.triggered.connect(lambda: self.createDataDialog("Picture1"))

        self.picture2Action = QtWidgets.QAction(self)
        self.picture2Action.setText("Picture2")
        self.picture2Action.triggered.connect(lambda: self.createDataDialog("Picture2"))

        self.trainAction = QtWidgets.QAction(self)
        self.trainAction.setText("Train")
        self.trainAction.triggered.connect(self.createTrainDialog)

        self.testSetAction = QtWidgets.QAction(self)
        self.testSetAction.setText("Testset")
        self.testSetAction.triggered.connect(lambda: self.testNetwork("Testset"))

        self.testPicAction = QtWidgets.QAction(self)
        self.testPicAction.setText("Picture")
        self.testPicAction.triggered.connect(lambda: self.testNetwork("Picture"))

        self.menuFile.addAction(self.loadNetworkAction)
        self.menuFile.addAction(self.saveNetworkAction)
        self.menuFile.addAction(self.ExportPyAction)

        self.menuLayer.addAction(self.convAction)
        self.menuLayer.addAction(self.linearAction)
        self.menuLayer.addAction(self.dropoutAction)

        self.menuFunctions.addAction(self.reluAction)
        self.menuFunctions.addAction(self.sigmoidAction)
        self.menuFunctions.addAction(self.maxPoolAction)
        self.menuFunctions.addAction(self.avgPoolAction)
        self.menuFunctions.addAction((self.viewAction))

        self.menuData.addAction(self.picture1Action)
        self.menuData.addAction(self.picture2Action)

        self.menuTest.addAction(self.testSetAction)
        self.menuTest.addAction(self.testPicAction)

        self.menuItems.addAction(self.menuFile.menuAction())
        self.menuItems.addAction(self.menuLayer.menuAction())
        self.menuItems.addAction(self.menuFunctions.menuAction())
        self.menuItems.addAction(self.menuData.menuAction())
        self.menuItems.addAction(self.trainAction)
        self.menuItems.addAction(self.menuTest.menuAction())

        self.localizeLanguage()
        # self.connect(self.data_thread, QtCore.SIGNAL("create_console_text"), self.create_console_text)
        self.dataThread.signal.connect(self.addLog)
        self.networkThread.signal.connect(self.addLog)
        self.networkThread.signal2.connect(self.saveNetworkAfterTrain)
        self.networkBuilder.signal.connect(self.randomizeWeights)
        QtCore.QMetaObject.connectSlotsByName(self)

    def localizeLanguage(self):
        localize = QtCore.QCoreApplication.translate
        self.setWindowTitle("NNBuilder")

    def scrollToBottom(self, minimum = None, maximum = None):
        self.scrollLog.verticalScrollBar().setValue(
            self.scrollLog.verticalScrollBar().maximum()
        )

    def mouseMoveEvent(self, event):
        if self.isResize:
            locate = event.pos() - self.mousePosition
            self.mousePosition = event.pos()
            self.scrollLog.setMaximumSize(self.scrollLog.size() - QtCore.QSize(0, locate.y()))

        elif (
                self.scrollNetworkBuilder.y() + self.scrollNetworkBuilder.height() + 30 <= event.y() <= self.scrollLog.y() + 30 and event.x() >= self.scrollNetworkBuilder.x() and
                event.x() <= self.scrollNetworkBuilder.x() + self.scrollNetworkBuilder.width()):
            self.setCursor(QtCore.Qt.SizeVerCursor)
        else:
            self.setCursor(QtCore.Qt.ArrowCursor)

    def mousePressEvent(self, event):
        if (event.y() >= self.scrollNetworkBuilder.y() + self.scrollNetworkBuilder.height() and
                event.y() <= self.scrollLog.y() + 30 and event.x() >= self.scrollNetworkBuilder.x() and
                event.x() <= self.scrollNetworkBuilder.x() + self.scrollNetworkBuilder.width()):
            self.mousePosition = event.pos()
            self.isResize = True

    def mouseReleaseEvent(self, event):
        self.isResize = False

    def buildLayer(self, type):
        layer = self.networkBuilder.network_layers
        connections = self.networkBuilder.connections
        firstLayer = self.networkBuilder.first_layer
        self.networkBuilder = NetworkBuilder(layer, connections, firstLayer)
        self.networkBuilder.AddLayer(type)
        self.scrollNetworkBuilder.setWidget(self.networkBuilder)

    def initLogger(self):
        for i in range(500):
            self.addLog(" ")

    def addLog(self, text):
        self.logger.add_text(text)
        self.logger.setMinimumSize(self.logger.size() + QtCore.QSize(0, 50))

    def clearLog(self):
        self.logger = Logger([])
        self.initLogger()
        self.scrollLog.setWidget(self.logger)

    def createDataDialog(self, type):
        self.dataThread.type = type
        self.dataThread.pathToTrain = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Training Directory"))
        self.dataThread.pathToTest = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Test Directory"))
        if (type == "Picture2"):
            self.dialogBox = DataLoadCfgDialog(True, True)
        else:
            self.dialogBox = DataLoadCfgDialog(True, False)
        self.dialogBox.acceptedButton.accepted.connect(self.loadData)

    def loadData(self):
        self.dataThread.batchSize = int(self.dialogBox.batchUpDown.value())
        self.dataThread.resize = int(self.dialogBox.resizeUpDown.value())
        if (self.dialogBox.normMeanInput.text() == "" and self.dialogBox.normStdInput.text() == ""):
            self.dataThread.transforms = transforms.Compose([transforms.Resize(self.dataThread.resize),
                                                             transforms.CenterCrop(self.dataThread.resize),
                                                             transforms.ToTensor(), ])
        else:
            self.dataThread.normalize = transforms.Normalize(
                mean=self.normalizePicture(self.dialogBox.normMeanInput.text()),
                std=self.normalizePicture(self.dialogBox.normStdInput.text())
            )

            self.dataThread.transforms = transforms.Compose([transforms.Resize(self.dataThread.resize),
                                                             transforms.CenterCrop(self.dataThread.resize),
                                                             transforms.ToTensor(),
                                                             self.dataThread.normalize])

        if (self.dataThread.type == "Picture2"):
            self.dataThread.categories = self.dialogBox.catInput.text().split(",")
        self.dialogBox.reject()
        self.dataThread.start()

    def normalizePicture(self, string):
        data = string.split(",")
        for i in range(len(data)):
            data[i] = float(data[i].strip(" "))
        return data

    def createTrainDialog(self):
        if (self.dataThread.trainData != []):
            self.dialogBox = Train_Dialog()
            self.dialogBox.buttonBox.accepted.connect(self.startTrain)
        else:
            self.addLog("Please, load data for start training")

    def loadNetwork(self):
        self.networkBuilder.connections = []
        self.networkBuilder.network_layers = []
        first = True
        fileDialog = QtWidgets.QFileDialog.getOpenFileName(self, "Load Network", "", "network (*.nx)")
        layerLoad = []
        if fileDialog != ('', ''):
            file = open(fileDialog[0], "r")
            for line in file:
                if first:
                    relatives = eval(line)[0]
                    first_layer = eval(line)[1]
                    pathToWeights = ""
                    if eval(line)[2] != "path of current weights":
                        for path in fileDialog[0].split("/")[:-1]:
                            pathToWeights += path + "/"
                        pathToWeights += eval(line)[2]
                    else:
                        pathToWeights = "path of current weights"

                    self.networkThread.pathToWeights = pathToWeights
                    self.networkThread.categories = eval(line)[3]
                    first = False
                else:
                    layerLoad.append(eval(line))

        for layer in layerLoad:
            self.buildLayer(layer[0][0])

        self.networkBuilder.set_layer_settings(relatives, layerLoad, first_layer)

    def saveNetworkBeforeTrain(self):
        saveLayers = self.networkBuilder.create_list_for_save()
        relatives = self.networkBuilder.connections
        firstLayer = self.networkBuilder.first_layer

        fileDialog = QtWidgets.QFileDialog.getSaveFileName(self, "Save Network", "", "network (*.nx)")
        if fileDialog != ('', ''):

            file = open(fileDialog[0], "w")

            file.write(str((relatives, firstLayer, self.networkThread.pathToWeights,
                            self.networkThread.categories)) + "\n")

            for layer in saveLayers:
                file.write(str(layer) + "\n")

            file.close()

    def saveNetworkAfterTrain(self, path):
        relatives = self.networkBuilder.connections
        firstLayer = self.networkBuilder.first_layer
        saveLayers = self.networkBuilder.create_list_for_save()

        fileDialog = path[:-3] + "_Network.nx"

        if fileDialog != ('', ''):
            file = open(fileDialog, "w")
            file.write(str((relatives, firstLayer, path.split("/")[-1], self.networkThread.categories)) + "\n")

            for layer in saveLayers:
                file.write(str(layer) + "\n")
            file.close()

    def randomizeWeights(self, nonsense):
        self.networkThread.pathToWeights = "path of current weights"

    def startTrain(self):
        self.loadTorchModel()
        self.networkThread.setOptimizer(self.dialogBox.optimizer_comboBox.currentText(), self.dialogBox.lr_line_edit.text(),
                                        self.dialogBox.momentum_line_edit.text())

        self.networkThread.setCriterion(self.dialogBox.loss_comboBox.currentText())

        self.networkThread.epochs = self.dialogBox.epochs_spinBox.value()

        self.networkThread.useGpu = self.dialogBox.gpu_radio_Button.isChecked()

        self.networkThread.trainData = self.dataThread.trainData
        self.networkThread.testData = self.dataThread.testData
        self.networkThread.batchSize = self.dataThread.batchSize

        self.networkThread.categories = self.dataThread.categories

        self.dialogBox.reject()
        self.networkThread.modelSavePath = QtWidgets.QFileDialog.getExistingDirectory()

        self.networkThread.start()

    def testNetwork(self, type):

        self.loadTorchModel()
        if type == "Testset":
            if self.dataThread.testData != []:
                self.networkThread.testData = self.dataThread.testData
                self.networkThread.batchSize = 64
                self.networkThread.test()
            else:
                self.addLog("Please, load data to test")

        elif type == "Picture":
            self.dialogBox = DataLoadCfgDialog(False, False)
            self.dialogBox.acceptedButton.accepted.connect(self.testNetworkByPic)

    def testNetworkByPic(self):

        if self.dialogBox.norm_mean_input.text() == "" and self.dialogBox.norm_std_input.text() == "":

            transmute = transforms.Compose([transforms.Resize(self.dialogBox.resize_spinBox.value()),
                                            transforms.CenterCrop(self.dialogBox.resize_spinBox.value()),
                                            transforms.ToTensor(), ])
        else:
            normalize = transforms.Normalize(
                mean=self.normalizePicture(self.dialogBox.norm_mean_input.text()),
                std=self.normalizePicture(self.dialogBox.norm_std_input.text())
            )

            transmute = transforms.Compose([transforms.Resize(self.dialogBox.resize_spinBox.value()),  # 256*256
                                            transforms.CenterCrop(self.dialogBox.resize_spinBox.value()),
                                            # schneidet im zentrum ab
                                            transforms.ToTensor(),
                                            normalize])
        fileName = QtWidgets.QFileDialog.getOpenFileName(self, "Select Picture", "C:", "format (*.jpg *.png)")

        image = Image.open(fileName[0])
        image_tensor = transmute(image)

        data = []
        data.append(image_tensor)

        data = torch.stack(data)

        if next(self.networkThread.model.parameters()).is_cuda:
            data = data.cuda()
        data = Variable(data)

        output = self.networkThread.model(data)
        predict = output.data.max(1, keepdim=True)[1].item()

        self.dialogBox = Test_Picture_Dialog(fileName[0], self.networkThread.categories[predict])

    def loadTorchModel(self):

        if self.networkThread.pathToWeights == "path of current weights":
            self.networkThread.model = Network(self.networkBuilder.get_network())
        else:
            if os.path.isfile(self.networkThread.pathToWeights):
                self.networkThread.model = torch.load(self.networkThread.pathToWeights, map_location=torch.device('cpu'))

    def ExportNetworkPy(self):
        fileDialog = QtWidgets.QFileDialog.getSaveFileName(self, "Export Network", "", "Python (*.py)")
        pathToWeights = fileDialog[0][:-3] + "_weights.pt"

        resize = self.dataThread.resize

        py_data = ["import torch", "from torchvision import transforms", "from PIL import Image",
                   "from torch.autograd import Variable", "import torch.nn.functional as F",
                   "import torch.nn as nn", " ", "path_weights = '" + str(pathToWeights) + "'",
                   "path_picture =  'Your Path here'", "resize = " + str(resize), " ",
                   "categories = " + str(self.networkThread.categories),
                   "transform = transforms.Compose([transforms.Resize(resize),",
                   "                                transforms.CenterCrop(resize),",
                   "                                transforms.ToTensor(), ])",
                   " ", "class Network(nn.Module): ", "    def __init__(self,layer_list):",
                   "        super(Network,self).__init__()", "        self.functions = []",
                   "        layers = []", "        for layer in layer_list:",
                   "            self.functions.append(self.create_layer(layer))",
                   "        for layer in self.functions:", "            if(not isinstance(layer, list)):",
                   "                layers.append(layer)", "        self.layer = nn.ModuleList(layers)",
                   " ", "    def forward(self, x):", "        layer_num = 0", "        for layer in self.functions:",
                   "            if(isinstance(layer, list)):", "                if(layer[0] == 'ReLu'):",
                   "                    x = F.relu(x)", "                elif(layer[0] == 'Sigmoid'):",
                   "                    x = torch.sigmoid(x)", "                elif(layer[0] == 'Max Pool'):",
                   "                    x = F.max_pool2d(x, layer[1])", "                elif(layer[0] == 'Avg Pool'):",
                   "                    x = F.avg_pool2d(x, layer[1])", "                elif(layer[0] == 'View'):",
                   "                    x = x.view(-1,layer[1])", "            else:",
                   "                x = self.layer[layer_num](x)", "                layer_num += 1", "        return x",
                   " ", "model = torch.load(path_weights)", " ", "img = Image.open(path_picture)",
                   "img_tensor = transform(img)", "data = []", "data.append(img_tensor)",
                   "data = torch.stack(data) #create Tensor([1,1,resize,resize])",
                   "if(next(model.parameters()).is_cuda):",
                   "    data = data.cuda()", "data = Variable(data)",
                   "out =  model(data) #network output", "prediction = out.data.max(1, keepdim=True)[1].item()",
                   "print(categories[prediction])"]

        if fileDialog != ('', ''):
            torch.save(self.networkThread.model, pathToWeights)

            file = open(fileDialog[0], "w")
            for line in py_data:
                file.write(line + "\n")
            file.close()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    StartWindow = GeneralWindow()
    StartWindow.showMaximized()
    sys.exit(app.exec_())
