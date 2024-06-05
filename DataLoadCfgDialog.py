from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator

class DataLoadCfgDialog(QtWidgets.QDialog):
    def __init__(self, hasBatch, isCatToSet):
        super().__init__()
        self.setWindowTitle("Data loading")
        self.resize(400, 279)
        font14 = QtGui.QFont()
        font14.setPointSize(14)
        font12 = QtGui.QFont()
        font12.setPointSize(12)

        # add button to window
        self.acceptedButton = QtWidgets.QDialogButtonBox(self)
        self.acceptedButton.setGeometry(QtCore.QRect(30, 240, 341, 32))
        self.acceptedButton.setOrientation(QtCore.Qt.Horizontal)
        self.acceptedButton.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.acceptedButton.setObjectName("acceptedButton")

        self.gridWidget = QtWidgets.QWidget(self)
        self.gridWidget.setGeometry(QtCore.QRect(9, 9, 391, 211))
        self.gridWidget.setObjectName("gridWidget")

        self.gridLayout = QtWidgets.QGridLayout(self.gridWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")

        #add resize input on window
        self.resizeInput = QtWidgets.QLabel(self.gridWidget)
        self.resizeInput.setFont(font14)
        self.resizeInput.setText("Resize:")
        self.resizeInput.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout.addWidget(self.resizeInput, 0, 0, 1, 1)
        #add up down for change value
        self.resizeUpDown = QtWidgets.QSpinBox(self.gridWidget)
        self.resizeUpDown.setFont(font12)
        self.resizeUpDown.setMinimum(1)
        self.resizeUpDown.setMaximum(9999999)
        self.gridLayout.addWidget(self.resizeUpDown, 0, 1, 1, 1)

        #add input for batch size
        self.batchInput = QtWidgets.QLabel(self.gridWidget)

        if(hasBatch):
            self.batchInput.setFont(font14)
            self.batchInput.setText("Batch size:")
            self.gridLayout.addWidget(self.batchInput, 1, 0, 1, 1, QtCore.Qt.AlignHCenter)

            self.batchUpDown = QtWidgets.QSpinBox(self.gridWidget)
            self.batchUpDown.setMinimum(1)
            self.batchUpDown.setMaximum(9999999)
            self.batchUpDown.setFont(font12)

            self.gridLayout.addWidget(self.batchUpDown, 1, 1, 1, 1)

        #add input for norm mean
        self.normMeanLabel = QtWidgets.QLabel(self.gridWidget)
        self.normMeanLabel.setFont(font14)
        self.normMeanLabel.setText("Normalize mean:")
        self.normMeanLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.gridLayout.addWidget(self.normMeanLabel, 2, 0, 1, 1)

        self.normMeanInput = QtWidgets.QLineEdit(self.gridWidget)
        self.normMeanInput.setFont(font12)
        reg_ex = QRegExp("^(0(\.\d{1,2})?|1(\.0{1,2})?)$")
        inputValidator = QRegExpValidator(reg_ex, self.normMeanInput)
        self.normMeanInput.setValidator(inputValidator)
        self.gridLayout.addWidget(self.normMeanInput, 2, 1, 1, 1)

        #add input for norm std
        self.normStdLabel = QtWidgets.QLabel(self.gridWidget)
        self.normStdLabel.setFont(font14)
        self.normStdLabel.setText("Normalize std:")
        self.normStdLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.gridLayout.addWidget(self.normStdLabel, 3, 0, 1, 1)

        self.normStdInput = QtWidgets.QLineEdit(self.gridWidget)
        self.normStdInput.setFont(font12)
        reg_ex = QRegExp("^(0(\.\d{1,2})?|1(\.0{1,2})?)$")
        inputValidator = QRegExpValidator(reg_ex, self.normStdInput)
        self.normStdInput.setValidator(inputValidator)
        self.normStdInput.setValidator(inputValidator)
        self.gridLayout.addWidget(self.normStdInput, 3, 1, 1, 1)

        #add input for categories
        if(isCatToSet):
            self.catLabel = QtWidgets.QLabel(self.gridWidget)
            self.catLabel.setFont(font14)
            self.catLabel.setText("Categories:")
            self.catLabel.setAlignment(QtCore.Qt.AlignCenter)

            self.gridLayout.addWidget(self.catLabel, 4, 0, 1, 1)

            self.catInput = QtWidgets.QLineEdit(self.gridWidget)
            self.catInput.setFont(font12)
            self.gridLayout.addWidget(self.catInput, 4, 1, 1, 1)


        self.acceptedButton.rejected.connect(self.reject)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.show()
