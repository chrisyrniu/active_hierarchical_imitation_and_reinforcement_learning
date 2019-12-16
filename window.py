from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot, QWaitCondition, QMutex, Qt
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QGridLayout
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush
import sys
import time
import worker
import numpy as np
class MainWindow(QWidget):
    mouse_sig = pyqtSignal(list)
    reset_sig = pyqtSignal()

    def __init__(self, vpos=None, vsize=None):
        super().__init__()
        self.pos_list = [[100.0,100.0],[200.0,200.0],[200.0,200.0]]
        self.width = 340
        self.init_x = 200
        self.init_y = 200
        self.envWidth = 17

        self.mouse_pos = [0,0]
        self.keys = {'A':0,'S0':1,'F':2}
        self.colors = [QColor(0,60,60),QColor(255,0,255),QColor(255,255,0)]
        self.mtx = QMutex()
        self.cond = QWaitCondition()
        vpos = vpos or [[8.25,0],[0,8.25],[-8.25,0],[0,-8.25],[0,0],[0,0],[7.25,0],[0,7.25],[-7.25,0],[0,-7.25]]
        self.flip(vpos)
        vsize = vsize or [[0.25,8.25],[8.5,0.25],[0.25,8.25],[8.5,0.25],[0.5,3.5],[3.5,0.5],[0.75,0.5],[0.5,0.75],[0.75,0.5],[0.5,0.75]]
        self.set_walls(vpos, vsize)
        self.initUI()
        self.setMouseTracking(True)

    def take_screenshot(self):
        self.update()
        screen = QApplication.primaryScreen().grabWindow(self.winId())
        screen.save('maze.png')

    def flip(self, pos_list):
        for v in pos_list:
            v[1] = -v[1]

    def set_walls(self, vpos, vsize):
        self.vpos = np.array(vpos,dtype=np.float32)
        self.vsize = np.array(vsize,dtype=np.float32)
        self.vpos = self.vpos-self.vsize
        self.vsize *= 2
        self.vsize *= self.width
        self.vsize /= self.envWidth
        self.vpos = self.width*(self.vpos+self.envWidth/2)/self.envWidth

    def initUI(self):
        self.setGeometry(self.init_x,self.init_y, self.width,self.width)
        self.setWindowTitle('Control Panel')
        self.show()

    def mousePressEvent(self, event):
        self.mouse_pos[0] = event.x()
        self.mouse_pos[1] = event.y()
        self.sendPos()
        self.cond.wakeAll()

    def mouseMoveEvent(self, event):
        self.mouse_pos[0] = event.x()
        self.mouse_pos[1] = event.y()
        self.sendPos()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R:
            self.resetEvent()

    @pyqtSlot()
    def sendPos(self):
        mouse_pos_flip = [0,0]
        mouse_pos_flip[0] = self.mouse_pos[0]
        mouse_pos_flip[1] = self.width - self.mouse_pos[1]
        self.mouse_sig.emit(mouse_pos_flip)

    @pyqtSlot()
    def resetEvent(self):
        self.reset_sig.emit()

    def onPosChange(self, pos_list):
        self.pos_list = pos_list.copy()
        self.update()

    def affine(self, xy):
        _xy = tuple(xy)
        _x = round(self.width*(_xy[0]+self.envWidth/2)/self.envWidth)
        _y = round(self.width*(_xy[1]+self.envWidth/2)/self.envWidth)
        return (_x,self.width-_y)

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.drawPoint(qp)
        brush = QBrush(Qt.gray, Qt.SolidPattern)
        assert(self.vpos.shape[0] == self.vsize.shape[0])
        qp.setBrush(brush)
        qp.setPen(Qt.NoPen)
        for i in range(self.vpos.shape[0]):
            qp.drawRect(round(self.vpos[i][0]), round(self.vpos[i][1]), round(self.vsize[i][0]),round(self.vsize[i][1]))

        qp.end()

    def drawPoint(self, qp):
        for i in range(len(self.pos_list)):
            qp.setPen(QPen(self.colors[i],15))
            _x,_y = self.affine(self.pos_list[i])
            qp.drawPoint(_x,_y)

    def myexit(self):
        self.close()
        QApplication.quit()
