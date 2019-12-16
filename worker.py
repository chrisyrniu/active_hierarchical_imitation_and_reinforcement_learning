# worker.py
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
import numpy as np


class Worker(QObject):
    over_sig = pyqtSignal()
    exit_sig = pyqtSignal()
    screen = pyqtSignal()
    pos_sig = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.pos_list = [[100.0,100.0],[1000.0,1000.0],[1000.0,1000.0]]
        self.in_pos = [100,100]
        self.mtx = None
        self.cond = None
        self.reset = False

    @pyqtSlot()
    def posChange(self): # A slot takes no params
        self.pos_sig.emit(self.pos_list)

    @pyqtSlot()
    def finished(self):
        self.over_sig.emit()

    def agentMove(self, x):
        self.pos_list[0] = x.tolist()

    def goalChange(self, pos, id):
        self.pos_list[1+id] = pos.tolist()

    def myWait(self):
        self.mtx.lock()
        try:
            self.cond.wait(self.mtx)
        finally:
            self.mtx.unlock()

    def setPos(self, pos):
        self.in_pos = pos.copy()

    def setReset(self):
        self.reset = True

    def unsetReset(self):
        self.reset = False

    def setCond(self, mtx, cond):
        self.mtx = mtx
        self.cond = cond

    def myexit(self):
        self.exit_sig.emit()

