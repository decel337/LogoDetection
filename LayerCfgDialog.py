from PyQt5 import QtCore, QtGui, QtWidgets

class LayerCfgDialog(QtWidgets.QDialog):
    def __init__(self, layerType):
        super().__init__()
        self.resize(399, 278)
        self.setWindowTitle("Configuration")
        self.acceptedButton = QtWidgets.QDialogButtonBox(self)
        self.acceptedButton.setObjectName("acceptedButton")
        self.acceptedButton.setGeometry(QtCore.QRect(31, 241, 340, 31))
        self.acceptedButton.setOrientation(QtCore.Qt.Horizontal)
        self.acceptedButton.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)

        self.gridWidget = QtWidgets.QWidget(self)
        self.gridWidget.setObjectName("gridWidget")
        self.gridWidget.setGeometry(QtCore.QRect(10, 10, 390, 210))

        self.gridLayout = QtWidgets.QGridLayout(self.gridWidget)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)

        font14 = QtGui.QFont()
        font14.setPointSize(14)

        font12 = QtGui.QFont()
        font12.setPointSize(12)

        if layerType in ["Max Pool", "Avg Pool"]:
            self.kernelLabel = QtWidgets.QLabel(self.gridWidget)
            self.kernelLabel.setFont(font14)
            self.kernelLabel.setText("Kernel size:")
            self.kernelLabel.setAlignment(QtCore.Qt.AlignCenter)

            self.gridLayout.addWidget(self.kernelLabel, 0, 0, 1, 1)
            self.kernelUpDown = QtWidgets.QSpinBox(self.gridWidget)
            self.kernelUpDown.setMinimum(1)
            self.kernelUpDown.setMaximum(9999999)
            self.kernelUpDown.setFont(font12)

            self.gridLayout.addWidget(self.kernelUpDown, 0, 1, 1, 1)
        else:
            self.inputLabel = QtWidgets.QLabel(self.gridWidget)
            self.inputLabel.setFont(font14)
            self.inputLabel.setText("Input size:")
            self.inputLabel.setAlignment(QtCore.Qt.AlignCenter)

            self.gridLayout.addWidget(self.inputLabel, 0, 0, 1, 1)

            self.inputUpDown = QtWidgets.QSpinBox(self.gridWidget)
            self.inputUpDown.setMinimum(1)
            self.inputUpDown.setMaximum(9999999)
            self.inputUpDown.setFont(font12)

            self.gridLayout.addWidget(self.inputUpDown, 0, 1, 1, 1)

            if layerType != "View":
                self.outputLabel = QtWidgets.QLabel(self.gridWidget)
                self.outputLabel.setFont(font14)
                self.outputLabel.setText("Output size:")
                self.gridLayout.addWidget(self.outputLabel, 1, 0, 1, 1, QtCore.Qt.AlignHCenter)

                self.outputUpDown = QtWidgets.QSpinBox(self.gridWidget)
                self.outputUpDown.setMinimum(1)
                self.outputUpDown.setMaximum(9999999)
                self.outputUpDown.setFont(font12)
                self.gridLayout.addWidget(self.outputUpDown, 1, 1, 1, 1)

            if layerType == "Convolution Layer":
                self.kernelLabel = QtWidgets.QLabel(self.gridWidget)
                self.kernelLabel.setFont(font14)
                self.kernelLabel.setText("Kernel size:")
                self.kernelLabel.setAlignment(QtCore.Qt.AlignCenter)

                self.gridLayout.addWidget(self.kernelLabel, 2, 0, 1, 1)

                self.kernelUpDown = QtWidgets.QSpinBox(self.gridWidget)
                self.kernelUpDown.setFont(font12)
                self.kernelUpDown.setMaximum(9999999)
                self.kernelUpDown.setMinimum(1)

                self.gridLayout.addWidget(self.kernelUpDown, 2, 1, 1, 1)

        self.acceptedButton.rejected.connect(self.reject)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.show()
