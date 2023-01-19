
from PySide6 import QtCore
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PySide6.QtUiTools import QUiLoader
from qdarktheme import load_stylesheet
import pyqtgraph as pg
import sys
import threading
import time, math
from numbersClassifier import NumbersClassifier
from numpy import random

class Graph:

    def __init__(self):
                
        self.app = QApplication(sys.argv)
        self.app.setStyleSheet(load_stylesheet())
        loader = QUiLoader()
        self.window = loader.load("tela.ui")
        self.progressBar = self.window.progressBar
        self.frame2 = self.window.frame2
        self.frame4 = self.window.frame4
        self.btn = self.window.btn_treinar

        self.b = pg.PlotWidget()
        self.i = pg.PlotWidget()
        self.t = 0
        
    def init(self,a,g):
        self.a = 0
        self.g = g
        self.create()
        self.window.show()
        sys.exit(self.app.exec())

    def create(self):   	
        self.frame2.layout().addWidget(self.b)
        self.frame4.layout().addWidget(self.i)

        pen1 = pg.mkPen(color=(255, 0, 0))
        d = self.i.plot(self.entr, self.said,pen=pen1)
        pen2 = pg.mkPen(color=(0, 0, 255))
        self.g1 = self.b.plot(self.entr, [0 for x in self.entr],pen=pen2)
        self.g2 = self.i.plot(self.entr, [0 for x in self.entr],pen=pen2)

    def draw(self):
        p = math.ceil(100*(self.a/self.g))
        self.progressBar.setValue(p)
	    	
    def plot(self, x, erro):
        #self.g1.plot([x for x in range(len(erro))], erro)
        self.g1.setData([x for x in range(len(erro))], erro)
        self.g2.setData(self.entr, x)