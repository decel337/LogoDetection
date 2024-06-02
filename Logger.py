from PyQt5 import QtCore, QtWidgets

class Logger(QtWidgets.QWidget):
    def __init__(self, listOfTexsts):
        super().__init__()
        self.texts = []
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.initializeConsole(listOfTexsts)
        self.setMouseTracking(True)

    def initializeConsole(self, listOfTexts):
        self.setObjectName("Logger")
        self.verticalLayout.setObjectName("loggerVerticalLayout")

        for text in listOfTexts:
            self.Write(text)

    def Write(self, text):
        self.texts.append(QtWidgets.QLabel())
        self.texts[-1].setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.texts[-1].setText(text)
        self.texts[-1].setMinimumSize(QtCore.QSize(1799, 21))
        self.verticalLayout.addWidget(self.texts[-1])

    def mouseMoveEvent(self, event):
        self.setCursor(QtCore.Qt.ArrowCursor)