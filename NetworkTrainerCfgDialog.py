from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator

class NetworkTrainerCfgDialog(QtWidgets.QDialog):

    def __init__(self):
        super().__init__()

        font14 = QtGui.QFont()
        font14.setPointSize(14)

        font12 = QtGui.QFont()
        font12.setPointSize(12)

        self.setWindowTitle("Train Settings")
        self.resize(400, 279)

        self.acceptedButton = QtWidgets.QDialogButtonBox(self)
        self.acceptedButton.setObjectName("acceptedButton")
        self.acceptedButton.setGeometry(QtCore.QRect(30, 240, 340, 30))
        self.acceptedButton.setOrientation(QtCore.Qt.Horizontal)
        self.acceptedButton.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)


        self.gridWidget = QtWidgets.QWidget(self)
        self.gridWidget.setObjectName("gridWidget")
        self.gridWidget.setGeometry(QtCore.QRect(10, 10, 390, 210))

        self.gridLayout = QtWidgets.QGridLayout(self.gridWidget)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)

        self.lossLabel = QtWidgets.QLabel(self.gridWidget)
        self.lossLabel.setFont(font14)
        self.lossLabel.setText("Loss function:")

        self.gridLayout.addWidget(self.lossLabel, 1, 0, 1, 1, QtCore.Qt.AlignHCenter)

        self.optimizerLabel = QtWidgets.QLabel(self.gridWidget)

        self.optimizerLabel.setFont(font14)
        self.optimizerLabel.setText("Optimizer:")
        self.optimizerLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.gridLayout.addWidget(self.optimizerLabel, 0, 0, 1, 1)

        self.optimizerComboBox = QtWidgets.QComboBox(self.gridWidget)
        self.optimizerComboBox.setFont(font12)
        self.optimizerComboBox.addItem("SGD")
        self.optimizerComboBox.addItem("Adam")

        self.gridLayout.addWidget(self.optimizerComboBox, 0, 1, 1, 1)

        self.lossComboBox = QtWidgets.QComboBox(self.gridWidget)
        self.lossComboBox.addItem("negativ log likelihood")
        self.lossComboBox.addItem("binary cross entropy")
        self.lossComboBox.setFont(font12)

        self.gridLayout.addWidget(self.lossComboBox, 1, 1, 1, 1)


        self.epochsLabel = QtWidgets.QLabel(self.gridWidget)
        self.epochsLabel.setFont(font14)
        self.epochsLabel.setText("Epochs:")
        self.epochsLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.gridLayout.addWidget(self.epochsLabel, 2, 0, 1, 1)

        self.epochsUpDown = QtWidgets.QSpinBox(self.gridWidget)
        self.epochsUpDown.setFont(font12)

        self.gridLayout.addWidget(self.epochsUpDown, 2, 1, 1, 1)

        self.learningRateLabel = QtWidgets.QLabel(self.gridWidget)
        self.learningRateLabel.setFont(font14)
        self.learningRateLabel.setText("Learning rate:")
        self.learningRateLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.gridLayout.addWidget(self.learningRateLabel, 3, 0, 1, 1)

        self.learningRateInput = QtWidgets.QLineEdit(self.gridWidget)
        reg_ex = QRegExp("^(0(\.\d{1,2})?|1(\.0{1,2})?)$")
        inputValidator = QRegExpValidator(reg_ex, self.learningRateInput)
        self.learningRateInput.setValidator(inputValidator)
        self.learningRateInput.setFont(font12)

        self.gridLayout.addWidget(self.learningRateInput, 3, 1, 1, 1)

        self.momentumLabel = QtWidgets.QLabel(self.gridWidget)
        self.momentumLabel.setFont(font14)
        self.momentumLabel.setText("Momentum:")
        self.momentumLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.gridLayout.addWidget(self.momentumLabel, 4, 0, 1, 1)

        self.momentumInput = QtWidgets.QLineEdit(self.gridWidget)
        reg_ex = QRegExp("^(0(\.\d{1,2})?|1(\.0{1,2})?)$")
        inputValidator = QRegExpValidator(reg_ex, self.momentumInput)
        self.momentumInput.setValidator(inputValidator)
        self.momentumInput.setFont(font12)

        self.gridLayout.addWidget(self.momentumInput, 4, 1, 1, 1)

        self.selectedGpu = QtWidgets.QRadioButton(self.gridWidget)
        self.selectedGpu.setFont(font14)
        self.selectedGpu.setText("GPU")

        self.gridLayout.addWidget(self.selectedGpu, 5, 1, 1, 1)

        self.acceptedButton.rejected.connect(self.reject)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.show()
