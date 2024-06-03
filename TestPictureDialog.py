from PyQt5 import QtCore, QtGui, QtWidgets

class TestPictureDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        font14 = QtGui.QFont()
        font14.setPointSize(14)

        font12 = QtGui.QFont()
        font12.setPointSize(12)

        self.setWindowTitle("Loading data")
        self.resize(400, 279)

        self.acceptedButton = QtWidgets.QDialogButtonBox(self)
        self.acceptedButton.setObjectName("acceptedButton")
        self.acceptedButton.setGeometry(QtCore.QRect(29, 241, 340, 31))
        self.acceptedButton.setOrientation(QtCore.Qt.Horizontal)
        self.acceptedButton.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)

        self.gridWidget = QtWidgets.QWidget(self)
        self.gridWidget.setObjectName("gridWidget")
        self.gridWidget.setGeometry(QtCore.QRect(10, 10, 390, 210))

        self.gridLayout = QtWidgets.QGridLayout(self.gridWidget)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)


        self.batchLabel = QtWidgets.QLabel(self.gridWidget)
        self.batchLabel.setFont(font14)
        self.batchLabel.setText("Batch size:")

        self.gridLayout.addWidget(self.batchLabel, 1, 0, 1, 1, QtCore.Qt.AlignHCenter)

        self.resizeLabel = QtWidgets.QLabel(self.gridWidget)
        self.resizeLabel.setFont(font14)
        self.resizeLabel.setText("Resize:")
        self.resizeLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.gridLayout.addWidget(self.resizeLabel, 0, 0, 1, 1)

        self.resizeUpDown = QtWidgets.QSpinBox(self.gridWidget)
        self.resizeUpDown.setFont(font12)

        self.gridLayout.addWidget(self.resizeUpDown, 0, 1, 1, 1)

        self.batchUpDown = QtWidgets.QSpinBox(self.gridWidget)
        self.batchUpDown.setFont(font12)

        self.gridLayout.addWidget(self.batchUpDown, 1, 1, 1, 1)

        self.normMeanLabel = QtWidgets.QLabel(self.gridWidget)
        self.normMeanLabel.setFont(font14)
        self.normMeanLabel.setText("Normalize mean:")
        self.normMeanLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.gridLayout.addWidget(self.normMeanLabel, 2, 0, 1, 1)

        self.normMeanInput = QtWidgets.QLineEdit(self.gridWidget)
        self.normMeanInput.setFont(font12)

        self.gridLayout.addWidget(self.normMeanInput, 2, 1, 1, 1)

        self.normStdLabel = QtWidgets.QLabel(self.gridWidget)
        self.normStdLabel.setFont(font14)
        self.normStdLabel.setText("Normalize std:")
        self.normStdLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.gridLayout.addWidget(self.normStdLabel, 3, 0, 1, 1)

        self.normStdInput = QtWidgets.QLineEdit(self.gridWidget)
        self.normStdInput.setFont(font12)
        self.gridLayout.addWidget(self.normStdInput, 3, 1, 1, 1)

        self.acceptedButton.rejected.connect(self.reject)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.show()

class TestPictureDialog(QtWidgets.QDialog):

    def __init__(self, path, pred):
        super().__init__()
        self.resize(400, 300)

        self.acceptedButton = QtWidgets.QDialogButtonBox(self)
        self.acceptedButton.setObjectName("acceptedButton")
        self.acceptedButton.setGeometry(QtCore.QRect(30, 240, 341, 32))
        self.acceptedButton.setOrientation(QtCore.Qt.Horizontal)
        self.acceptedButton.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)

        self.verticalWidget = QtWidgets.QWidget(self)
        self.verticalWidget.setObjectName("verticalWidget")
        self.verticalWidget.setGeometry(QtCore.QRect(19, 9, 371, 191))

        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalWidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)

        self.pic = QtWidgets.QLabel(self.verticalWidget)
        self.pic.setText("Picture")

        self.pxlmap = QtGui.QPixmap(path)
        self.pic.setPixmap(self.pxlmap)

        self.verticalLayout.addWidget(self.pic, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

        self.predictImage = QtWidgets.QLabel(self.verticalWidget)
        self.predictImage.setText("Prediction: " + str(pred))

        self.verticalLayout.addWidget(self.predictImage, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)

        self.acceptedButton.accepted.connect(self.accept)
        self.acceptedButton.rejected.connect(self.reject)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.show()
