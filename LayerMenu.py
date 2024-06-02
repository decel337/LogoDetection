from PyQt5 import QtWidgets

class LayerMenu(QtWidgets.QMenu):
    def __init__(self, layerId, isEdit):
        super().__init__()
        self.layerId = layerId

        self.editAction = QtWidgets.QAction(self)
        self.editAction.setText("Change config")
        self.drawAction = QtWidgets.QAction(self)
        self.drawAction.setText("Create Relative")
        self.deleteAction = QtWidgets.QAction(self)
        self.deleteAction.setText("Remove")
        self.DeleteRelativeAction = QtWidgets.QAction(self)
        self.DeleteRelativeAction.setText("Delete Relative")
        self.SetFirstLayerAction = QtWidgets.QAction(self)
        self.SetFirstLayerAction.setText("Set first layer")

        if isEdit:
            self.addAction(self.editAction)

        self.addAction(self.drawAction)
        self.addAction(self.deleteAction)
        self.addAction(self.DeleteRelativeAction)
        self.addAction(self.SetFirstLayerAction)
