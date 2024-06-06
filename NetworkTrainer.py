from PyQt5 import QtCore

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class NetworkTrainer(QtCore.QThread):
    signal = QtCore.pyqtSignal('PyQt_PyObject')
    signal2 = QtCore.pyqtSignal('PyQt_PyObject')

    def __init__(self):
        QtCore.QThread.__init__(self)
        self.model = "Network Object"
        self.optimizer = "optimizer"
        self.criterion = "loss function"
        self.testData = "test data"
        self.trainData = "train data"
        self.modelSavePath = "weights folder"
        self.pathToWeights = "path of current weights"
        self.categories = "list of categories for prediction"
        self.batchSize = "batch_size"

        self.epochs = 0
        self.useGpu = False

    def setOptimizer(self, type, lr, mom):
        if lr == "":
            lr = "0.8"
        if type == "SGD":
            if mom != "":
                self.optimizer = optim.SGD(self.model.parameters(), lr=float(lr), momentum=float(mom))
            else:
                self.optimizer = optim.SGD(self.model.parameters(), lr=float(lr))

        elif type == "Adam":
            if lr != "":
                self.optimizer = optim.Adam(self.model.parameters(), lr=float(lr))

    def setCriterion(self, type):
        if type == "negativ log likelihood":
            self.criterion = F.nll_loss
        elif type == "binary cross entropy":
            self.criterion = F.binary_cross_entropy

    def train(self, epoch):
        if torch.cuda.is_available() and self.useGpu:
            self.model.cuda()
        self.model.train()
        batchId = 0
        for data, target in self.trainData:
            if torch.cuda.is_available() and self.useGpu:
                data = data.cuda()
            target = torch.Tensor(target)
            if torch.cuda.is_available() and self.useGpu:
                target = target.cuda()
            try:
                data = Variable(data)
                target = Variable(target)
                self.optimizer.zero_grad()
                out = self.model(data)
                criterion = self.criterion
                loss = criterion(out, target)
                loss.backward()
                self.optimizer.step()
                self.signal.emit('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batchId * len(data), len(self.trainData) * self.batchSize,
                           100. * batchId / len(self.trainData), loss.data))
            except Exception as ex:
                self.signal.emit(f'Error train: {ex}')

            batchId += + 1

        self.pathToWeights = self.modelSavePath + "/weights_" + "epoch_" + str(epoch) + '.pt'
        torch.save(self.model, self.pathToWeights)

        self.signal2.emit(self.pathToWeights)

    def test(self):
        self.model.eval()
        loss = 0
        correct = 0
        success = True
        incorrect = 0
        predictions = 0

        if len(self.testData) == 0:
            self.signal.emit(f'Not load data for test')

        for data, target in self.testData:
            if torch.cuda.is_available() and self.useGpu:
                data = data.cuda()
            data = Variable(data, volatile=True)
            target = torch.Tensor(target)
            if torch.cuda.is_available() and self.useGpu:
                target = target.cuda()

            try:
                target = Variable(target)
                out = self.model(data)

                loss += F.binary_cross_entropy(out, target, size_average=False).data
                prediction = out.data.max(1, keepdim=True)[1]
                predictions += len(prediction)
                for i in range(len(prediction)):
                    correct += 1 if (target[i][prediction[i].item()].item() == 1) else 0
                    incorrect += 0 if (target[i][prediction[i].item()].item() == 1) else 1
            except Exception as error:
                self.signal.emit(f'Error test: {error}')
                success = False

        if success:
            loss = loss / len(self.testData) * self.batchSize
            self.signal.emit('Average loss: ' + str(round(loss.item(), 6)))
            self.signal.emit("Accuracy: " + str(round(100 * correct / (len(self.testData) * self.batchSize), 2)) + " %")

    def run(self):
        for epoch in range(self.epochs):
            self.train(epoch)
            self.test()
