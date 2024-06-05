from PyQt5 import QtCore, QtGui, QtWidgets

class Layer(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.initializeLayer()

    def initializeLayer(self):
        self.setMinimumSize(299, 101)
        self.widget = QtWidgets.QWidget(self)
        self.widget.setObjectName("widget")
        self.widget.setMinimumSize(299, 101)
        self.widget.setStyleSheet("background-color: rgb(211, 211, 211);\n"
                                  "border: 1px solid black;")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.labelName = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.labelName.setFont(font)
        self.labelName.setText("default")
        self.verticalLayout.addWidget(self.labelName, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.layerCfg = QtWidgets.QLabel(self.widget)
        self.layerCfg.setText("in: 10000 out: 30000 kernel: 500x500")
        font = QtGui.QFont()
        font.setPointSize(10)
        self.layerCfg.setFont(font)
        self.verticalLayout.addWidget(self.layerCfg, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


class LinearLayer(Layer):
    def __init__(self, parent):
        super().__init__(parent)
        self.initializeLayer()
        self.inputSize = 50
        self.outputSize = 200
        self.type = "Linear Layer"
        self.labelName.setText(self.type)
        self.updateSettings()

    def updateSettings(self):
        self.layerCfg.setText("in: " + str(self.inputSize) + " out: " + str(self.outputSize))


class ConvolutionLayer(Layer):
    def __init__(self, parent):
        super().__init__(parent)
        self.initializeLayer()
        self.type = "Convolution Layer"
        self.inputSize = 50
        self.outputSize = 200
        self.kernelSize = 50
        self.labelName.setText(self.type)
        self.updateSettings()

    def updateSettings(self):
        self.layerCfg.setText("in: " + str(self.inputSize) + " out: " + str(self.outputSize) +
                              " kernel: " + str(self.kernelSize) + "x" + str(self.kernelSize))


class PoolingLayer(Layer):
    def __init__(self, parent, type):
        super().__init__(parent)
        self.initializeLayer()
        self.type = type
        self.kernelSize = 5
        self.labelName.setText(self.type)
        self.updateSettings()

    def updateSettings(self):
        self.layerCfg.setText("kernel: " + str(self.kernelSize) + "x" + str(self.kernelSize))


class ViewLayer(Layer):
    def __init__(self, parent):
        super().__init__(parent)
        self.initializeLayer()
        self.type = "View"
        self.inputSize = 100
        self.labelName.setText(self.type)
        self.updateSettings()

    def updateSettings(self):
        self.layerCfg.setText("in/out: " + str(self.inputSize))


class FunctionLayer(QtWidgets.QWidget):
    def __init__(self, parent, type):
        super().__init__(parent)
        self.type = type
        self.initializeLayer()

    def initializeLayer(self):
        self.setMinimumSize(200, 60)
        self.widget = QtWidgets.QWidget(self)
        self.widget.setStyleSheet("background-color: rgb(211, 211, 211);\n"
                                  "border: 1px solid black;")
        self.widget.setMinimumSize(200, 60)

        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.labelName = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.labelName.setFont(font)
        self.labelName.setAlignment(QtCore.Qt.AlignCenter)
        self.labelName.setText(self.type)
        self.verticalLayout.addWidget(self.labelName)
