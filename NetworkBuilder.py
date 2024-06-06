from PyQt5 import QtCore, QtGui, QtWidgets
from Layer import LinearLayer, FunctionLayer, ViewLayer, ConvolutionLayer, PoolingLayer
from LayerMenu import LayerMenu
from LayerCfgDialog import LayerCfgDialog
import random


class NetworkBuilder(QtWidgets.QWidget):
    signal = QtCore.pyqtSignal('PyQt_PyObject')

    def __init__(self, layerList, relativesList, firstLayer):
        super().__init__()
        self.layers = layerList
        self.relatives = relativesList
        self.firstLayer = firstLayer
        self.prevMousePos = -1
        self.mousePos = 0
        self.posX = 0
        self.posY = 0
        self.drawStart = -1
        self.changeLayerIdx = 0
        self.dialog = 0
        self.menu = 0
        self.setMouseTracking(True)
        self.InitializeNetworkBuilder()

    def InitializeNetworkBuilder(self):
        self.setObjectName("Network Builder")
        self.setGeometry(QtCore.QRect(0, 0, 50000, 50000))
        self.setMinimumSize(QtCore.QSize(50000, 50000))

        for layer in self.layers:
            layer.setParent(self)

    def getLayer(self, layerIdx):
        if self.layers[layerIdx].type == "Linear Layer":
            return self.layers[layerIdx].type, self.layers[layerIdx].inputSize, self.layers[layerIdx].outputSize
        elif self.layers[layerIdx].type == "Convolution Layer":
            return self.layers[layerIdx].type, self.layers[layerIdx].inputSize, self.layers[layerIdx].outputSize, \
            self.layers[layerIdx].kernelSize
        elif self.layers[layerIdx].type in ["Max Pool", "Avg Pool"]:
            return self.layers[layerIdx].type, self.layers[layerIdx].kernelSize
        elif self.layers[layerIdx].type == "View":
            return self.layers[layerIdx].type, self.layers[layerIdx].inputSize
        else:
            return self.layers[layerIdx].type, 0

    def getNetwork(self):
        currentLayer = self.firstLayer
        network = []
        isConnected = True
        while isConnected:
            isConnected = False
            network.append(self.getLayer(currentLayer))
            for relative in self.relatives:
                if relative[0] == currentLayer:
                    currentLayer = relative[1]
                    isConnected = True
                    break
        return network

    def saveModel(self):
        save_list = []
        for i in range(len(self.layers)):
            settings = self.getLayer(i)
            geometrie = (self.layers[i].x(), self.layers[i].y())
            save_list.append((settings, geometrie))
        return save_list

    def setLayerCfg(self, relatives, listForLoad, firstLayer):
        self.relatives = relatives
        self.setFirstLayer(firstLayer)
        for i in range(len(self.layers)):
            if self.layers[i].type == "Linear Layer":
                self.layers[i].inputSize = listForLoad[i][0][1]
                self.layers[i].outputSize = listForLoad[i][0][2]
            elif self.layers[i].type == "Convolution Layer":
                self.layers[i].inputSize = listForLoad[i][0][1]
                self.layers[i].outputSize = listForLoad[i][0][2]
                self.layers[i].kernelSize = listForLoad[i][0][3]
            elif self.layers[i].type in ["Max Pool", "Avg Pool"]:
                self.layers[i].kernelSize = listForLoad[i][0][1]
            elif self.layers[i].type == "View":
                self.layers[i].inputSize = listForLoad[i][0][1]
            self.layers[i].move(listForLoad[i][1][0], listForLoad[i][1][1])
            if not isinstance(self.layers[i], FunctionLayer):
                self.layers[i].updateSettings()

    def addLayer(self, layer):
        if layer == "Linear Layer":
            self.layers.append(LinearLayer(self))
        elif layer == "Convolution Layer":
            self.layers.append(ConvolutionLayer(self))
        elif layer in ["Max Pool", "Avg Pool"]:
            self.layers.append(PoolingLayer(self, layer))
        elif layer == "View":
            self.layers.append(ViewLayer(self))
        else:
            self.layers.append(FunctionLayer(self, layer))
        self.layers[-1].move(50, 50)

        if self.firstLayer == -1:
            self.firstLayer = len(self.layers) - 1

            self.layers[len(self.layers) - 1].widget.setStyleSheet("background-color: rgb(0, 204, 51);\n"
                                                                   "border: 1px solid black;")
            self.update()

    def mousePressEvent(self, event):
        if event.button() == 1:
            for i in range(len(self.layers)):
                if (self.checkOverlap(i, event)):
                    self.setEndRelative(i)
                    self.prevMousePos = i
                    self.mousePos = event.pos()
                    self.setCursor(QtCore.Qt.ClosedHandCursor)
                    break

        elif event.button() == 2:
            for i in range(len(self.layers)):
                if self.checkOverlap(i, event):
                    self.showMenu(event, i)
                    break

    def createRelative(self, layerIdx):
        if self.drawStart != -1:
            self.setEndRelative(layerIdx)
        else:
            self.setStartRelative(layerIdx)

    def setStartRelative(self, layerIdx):
        if self.checkRelatives(layerIdx, 0):
            self.drawStart = layerIdx

    def setEndRelative(self, layerIdx):
        if self.drawStart not in [layerIdx, -1]:
            if self.checkRelatives(layerIdx, 1):
                self.relatives.append((self.drawStart, layerIdx))
        self.drawStart = -1
        self.update()

    def showMenu(self, event, layerIdx):
        isEdit = not isinstance(self.layers[layerIdx], FunctionLayer)
        self.menu = LayerMenu(layerIdx, isEdit)
        if (isEdit):
            self.menu.editAction.triggered.connect(lambda: self.showLayerDialog(layerIdx))
        self.menu.drawAction.triggered.connect(lambda: self.setStartRelative(layerIdx))
        self.menu.deleteAction.triggered.connect(lambda: self.removeLayer(layerIdx))
        self.menu.DeleteRelativeAction.triggered.connect(lambda: self.removeRelative(layerIdx))
        self.menu.SetFirstLayerAction.triggered.connect(lambda: self.setFirstLayer(layerIdx))
        action = self.menu.exec_(self.mapToGlobal(event.pos()))

    def setFirstLayer(self, layerIdx):
        self.layers[self.firstLayer].widget.setStyleSheet("background-color: rgb(211, 211, 211);\n"
                                                          "border: 1px solid black;")

        self.firstLayer = layerIdx
        self.layers[layerIdx].widget.setStyleSheet("background-color: rgb(0, 204, 51);\n"
                                                   "border: 1px solid black;")

        for relative in self.relatives:
            if relative[1] == layerIdx:
                self.relatives.remove(relative)

    def mouseReleaseEvent(self, event):
        self.prevMousePos = -1
        self.setCursor(QtCore.Qt.OpenHandCursor)

    def mouseMoveEvent(self, event):
        if self.prevMousePos != -1:
            distance = event.pos() - self.mousePos
            self.mousePos = event.pos()
            self.layers[self.prevMousePos].move(self.layers[self.prevMousePos].pos() + distance)
            self.update()
        elif self.drawStart != -1:
            self.posX = event.x()
            self.posY = event.y()
            self.update()
        else:
            self.setCursor(QtCore.Qt.OpenHandCursor)

    def mouseDoubleClickEvent(self, event):
        for i in range(len(self.layers)):
            if self.checkOverlap(i, event):
                self.createRelative(i)
                break

    def showLayerDialog(self, i):
        self.changeLayerIdx = i
        self.dialog = LayerCfgDialog(self.layers[i].type)
        self.dialog.acceptedButton.accepted.connect(self.changeLayer)

    def paintEvent(self, event):
        drawer = QtGui.QPainter(self)
        drawer.setPen(QtGui.QColor(0, 0, 0))
        drawer.setFont(QtGui.QFont("Arial", 400))
        if self.drawStart >= len(self.layers):
            return
        if self.drawStart != -1:
            drawer.drawLine(self.layers[self.drawStart].x() + self.layers[self.drawStart].width() /
                             2, self.layers[self.drawStart].y() + self.layers[self.drawStart].height(),
                             self.posX, self.posY)
        for relative in self.relatives:
            if relative[0] >= len(self.layers) or relative[1] >= len(self.layers):
                continue
            drawer.drawLine(self.layers[relative[0]].x() + self.layers[relative[0]].width() / 2,
                             self.layers[relative[0]].y() + self.layers[relative[0]].height(),
                             self.layers[relative[1]].x() + self.layers[relative[1]].width() / 2,
                             self.layers[relative[1]].y())

    def checkOverlap(self, layerIdx, event):
        if (event.x() >= self.layers[layerIdx].x() - 5 and event.x() - 5 <= self.layers[layerIdx].x()
                + self.layers[layerIdx].width() and event.y() >= self.layers[layerIdx].y() - 5
                and event.y() - 5 <= self.layers[layerIdx].y() + self.layers[layerIdx].height()):
            return True
        else:
            return False

    def checkRelatives(self, layerIdx, index):
        if layerIdx == self.firstLayer and index == 1:
            return False
        for relative in self.relatives:
            if relative[index] == layerIdx:
                return False
        return True

    def changeLayer(self):
        if self.layers[self.changeLayerIdx].type not in ["Max Pool", "Avg Pool"]:
            self.layers[self.changeLayerIdx].inputSize = int(self.dialog.inputUpDown.value())
            if self.layers[self.changeLayerIdx].type != "View":
                self.layers[self.changeLayerIdx].outputSize = int(self.dialog.outputUpDown.value())
        if self.layers[self.changeLayerIdx].type in ["Convolution Layer", "Max Pool", "Avg Pool"]:
            self.layers[self.changeLayerIdx].kernelSize = int(self.dialog.kernelUpDown.value())

        self.layers[self.changeLayerIdx].updateSettings()
        self.dialog.reject()
        self.update()

    def removeRelative(self, layerIdx):
        forRemove = []
        for i in range(len(self.relatives)):
            if layerIdx in self.relatives[i]:
                forRemove.append(i - len(forRemove))
        for index in forRemove:
            self.relatives.pop(index)

    def removeLayer(self, layerIdx):
        self.layers[layerIdx].deleteLater()
        self.layers.pop(layerIdx)

        self.removeRelative(layerIdx)

        newRelatives = []
        for relative in self.relatives:
            start = relative[0] if (relative[0] < layerIdx) else relative[0] - 1
            end = relative[1] if (relative[1] < layerIdx) else relative[1] - 1
            newRelatives.append((start, end))
        if self.firstLayer >= layerIdx:
            self.firstLayer = (-1 if not self.layers else random.randint(0, len(self.layers)-1)) if (self.firstLayer == layerIdx) else self.firstLayer - 1
            if self.firstLayer != - 1:
                self.setFirstLayer(self.firstLayer)
        self.relatives = newRelatives
