#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import os, sys
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QWidget, QApplication, QHBoxLayout, QVBoxLayout, QSizePolicy, QSpacerItem, QComboBox,
                             QPushButton, QMessageBox,QTextEdit)
from PyQt5.QtGui import QIcon, QFont
# from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import pyqtSignal
import seaborn as sns
import pandas as pd
# from sklearn.neighbors.kde import KernelDensity
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy import stats
import numpy as np
import qdarkstyle
import InputDialog
import threading
import random
import py3Dmol
plt.style.use('ggplot')

""" Seaborn """
class ClusteringDiagram(QWidget):
    def __init__(self):
        super(ClusteringDiagram, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.layout = QVBoxLayout(self)

    def seabornPlot(self):
        g = sns.FacetGrid(self.df, hue="time", palette="Set1",
                          hue_order=["Dinner", "Lunch", "Dinner1", "Dinner2", "Dinner3", "Dinner4", "Dinner5",
                                     "Dinner6", "Dinner7", "Dinner8", "Dinner9"])
        g.map(plt.scatter, "total_bill", "tip", s=50, edgecolor="w", norm=0)
        g.add_legend()
        return g.fig

    def initPlot(self, df):
        self.df = df
        fig = self.seabornPlot()
        figureCanvas = FigureCanvas(fig)
        figureCanvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        figureCanvas.updateGeometry()
        navigationToolbar = NavigationToolbar2QT(figureCanvas, self)
        control_layout = QHBoxLayout()
        control_layout.addStretch(1)
        control_layout.addWidget(navigationToolbar)
        control_layout.addStretch(1)
        self.layout.addWidget(figureCanvas)
        self.layout.addLayout(control_layout)
        self.show()

""" Matplotlib """
class MyFigureCanvas(FigureCanvas):
    """ Canvas """

    def __init__(self):
        self.figure = Figure()
        super().__init__(self.figure)

class MyFigureCanvas1(QTextEdit):  # 使用 QTextEdit 来显示 Py3Dmol 的内容
    pass
# Cluster diagram
class ClusteringDiagramMatplotlib(QWidget):
    def __init__(self):
        super(ClusteringDiagramMatplotlib, self).__init__()
        self.fontdict = {
            'family': 'Arial',
            'size': 16,
            'color': '#282828',
        }
        self.colorlist = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628', '#F781BF',
                          '#999999']
        self.marklist = ['o'] * 9 + ['v'] * 9 + ['^'] * 9 + ['+'] * 9
        self.initUI()

    def initUI(self):
        self.setWindowTitle('iLearnPlus Scatter plot')
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.figureCanvas = MyFigureCanvas()
        layout.addWidget(self.figureCanvas)
        self.navigationToolbar = NavigationToolbar2QT(self.figureCanvas, self)
        hLayout = QHBoxLayout()
        hLayout.addStretch(1)
        hLayout.addWidget(self.navigationToolbar)
        hLayout.addStretch(1)
        layout.addLayout(hLayout)

    def init_data(self, method, data, xlabel=None, ylabel=None):
        self.xlabel = xlabel if not xlabel is None else 'PC 1'
        self.ylabel = ylabel if not ylabel is None else 'PC 2'
        self.method = method
        self.prefix = 'Cluster:' if method == 'Clustering' else 'Sample category:'
        self.data = data
        self.fig = self.figureCanvas.figure.add_subplot(111)
        self.__draw_figure__()

    def __draw_figure__(self):
        self.fig.cla()
        self.fig.set_facecolor('white')
        self.fig.set_title(self.method)
        self.fig.set_xlabel(self.xlabel, fontdict=self.fontdict)
        self.fig.set_ylabel(self.ylabel, fontdict=self.fontdict)
        for i, item in enumerate(self.data):
            self.fig.scatter(item[1][:, 0], item[1][:, 1], color=self.colorlist[i % len(self.colorlist)], s=70,
                             marker=self.marklist[i % len(self.marklist)], label='%s %s' % (self.prefix, item[0]),
                             edgecolor="w")
        self.fig.legend()
        labels = self.fig.get_xticklabels() + self.fig.get_yticklabels()
        [label.set_color('#282828') for label in labels]
        [label.set_fontname('Arial') for label in labels]
        [label.set_size(16) for label in labels]
        self.fig.spines['left'].set_color('black')
        self.fig.spines['bottom'].set_color('black')

# Histogram
class HistogramWidget(QWidget):
    def __init__(self):
        super(HistogramWidget, self).__init__()
        self.fontdict = {
            'family': 'Arial',
            'size': 16,
            'color': '#282828',
        }
        self.color =['#FF0099', '#008080', '#FF9900', '#660033']
        self.initUI()

    def initUI(self):
        self.setWindowTitle('iLearnPlus Histogram')
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.figureCanvas = MyFigureCanvas()
        layout.addWidget(self.figureCanvas)
        self.navigationToolbar = NavigationToolbar2QT(self.figureCanvas, self)
        hLayout = QHBoxLayout()
        hLayout.addStretch(1)
        hLayout.addWidget(self.navigationToolbar)
        hLayout.addStretch(1)
        layout.addLayout(hLayout)

    def init_data(self, title, data):        
        self.title = title
        # if data is too large, random part samples
        if data.shape[0] * (data.shape[1]-1) > 32000:
            sel_num = int(32000 / (data.shape[1]-1))
            random_index = random.sample(range(0, len(data)), sel_num)
            self.data = data[random_index]
        else:
            self.data = data
        self.categories = sorted(set(self.data[:, 0]))
        self.fig = self.figureCanvas.figure.add_subplot(111)
        self.__draw_figure__()

    def __draw_figure__(self):
        try:
            self.fig.cla()
            self.fig.set_title(self.title)
            self.fig.set_xlabel("Value bins", fontdict=self.fontdict)
            self.fig.set_ylabel("Density", fontdict=self.fontdict)
            max_value = max(self.data[:, 1:].reshape(-1))
            min_value = min(self.data[:, 1:].reshape(-1))
            if max_value == min_value and max_value == 0:
                pass
            else:
                bins = np.linspace(min_value, max_value, 10)
                for i, c in enumerate(self.categories):
                    tmp_data = self.data[np.where(self.data[:, 0]==c)][:, 1:].reshape(-1)
                    self.fig.hist(tmp_data, bins=bins, stacked=True, density=True, facecolor=self.color[i%len(self.color)], alpha=0.5)
                    X_plot = np.linspace(min_value, max_value, 100)[:, np.newaxis]
                    bandwidth = (max_value - min_value) / 20.0
                    if bandwidth <= 0:
                        bandwidth = 0.1
                    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(tmp_data.reshape((-1, 1)))
                    log_dens = kde.score_samples(X_plot)
                    self.fig.plot(X_plot[:, 0], np.exp(log_dens), color=self.color[i%len(self.color)], label='Category %s' %int(c))
                self.fig.legend()        
            labels = self.fig.get_xticklabels() + self.fig.get_yticklabels()
            [label.set_color('#282828') for label in labels]
            [label.set_fontname('Arial') for label in labels]
            [label.set_size(16) for label in labels]
        except Exception as e:
            QMessageBox.critical(self, 'Plot error', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

class Py3DmolWidget(QWidget):
    def __init__(self):
        super(Py3DmolWidget, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Py3Dmol')
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.figureCanvas = MyFigureCanvas1()
        layout.addWidget(self.figureCanvas)
        # self.navigationToolbar = NavigationToolbar2QT(self.figureCanvas, self)
        # hLayout = QHBoxLayout()
        # hLayout.addStretch(1)
        # hLayout.addWidget(self.navigationToolbar)
        # hLayout.addStretch(1)
        # layout.addLayout(hLayout)
        # self.init_data(pdb_data=self.pdb_str)

    def init_data(self, pdb_data=None):
        self.pdb_str = pdb_data
        self.__draw_figure__()

    def __draw_figure__(self):
        # Clear the previous view
        if hasattr(self, 'view'):
            self.view.clear()

        # Create a new view
        self.view = py3Dmol.view()
        self.view.addModel(self.pdb_str, 'pdb')
        self.view.setStyle({'cartoon': {'color': 'white'}})
        self.view.setStyle({'chain': 'A'}, {'cartoon': {'color': 'red'}})
        self.view.show()


# ROC and PRC curve
class CurveWidget(QWidget):
    def __init__(self):
        super(CurveWidget, self).__init__()
        self.colorlist = ['#377EB8', '#FF1493', '#984EA3', '#FF7F00', '#A65628', '#F781BF', '#999999', '#4DAF4A',
                          '#D2691E', '#DEB887']
        self.fontdict = {
            'family': 'Arial',
            'size': 16,
            'color': '#282828',
        }
        self.initUI()

    def initUI(self):
        self.setWindowTitle('iLearnPlus Curve')
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.figureCanvas = MyFigureCanvas()
        layout.addWidget(self.figureCanvas)
        self.navigationToolbar = NavigationToolbar2QT(self.figureCanvas, self)
        hLayout = QHBoxLayout()
        hLayout.addStretch(1)
        hLayout.addWidget(self.navigationToolbar)
        hLayout.addStretch(1)
        layout.addLayout(hLayout)

    def init_data(self, type, title, fold_data=None, mean_data=None, ind_data=None):
        self.title = title
        self.fold_data = fold_data
        self.mean_data = mean_data
        self.ind_data = ind_data
        self.type = type
        if type == 0:
            self.x = 'fpr'
            self.y = 'tpr'
        else:
            self.x = 'recall'
            self.y = 'precision'
        self.fig = self.figureCanvas.figure.add_subplot(111)
        self.__draw_figure__()

    def __draw_figure__(self):
        self.fig.cla()
        self.fig.set_facecolor('white')
        self.fig.set_xlim((-0.05, 1.05))
        self.fig.set_ylim((-0.05, 1.05))
        self.fig.set_title(self.title)
        if self.title == 'ROC curve':
            x_label = 'False positive rate'
            y_label = 'True positive rate'
        else:
            x_label = 'Recall'
            y_label = 'Precision'
        self.fig.set_xlabel(x_label, fontdict=self.fontdict)
        self.fig.set_ylabel(y_label, fontdict=self.fontdict)
        self.fig.set_xticks(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        self.fig.set_yticks(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        if not self.fold_data is None:
            for i, item in enumerate(self.fold_data):
                self.fig.plot(item[1][self.x], item[1][self.y], label=item[0], color=self.colorlist[i%len(self.colorlist)], lw=2, alpha=1.0)
        if not self.mean_data is None:
            self.fig.plot(self.mean_data[1][self.x], self.mean_data[1][self.y], label=self.mean_data[0], lw=2,
                          alpha=0.8, color='b')
        if not self.ind_data is None:
            self.fig.plot(self.ind_data[1][self.x], self.ind_data[1][self.y], label=self.ind_data[0], lw=2, alpha=0.5,
                          color='r')
        if self.type == 0:
            self.fig.plot([0, 1], [0, 1], label='Random', lw=2, alpha=0.5, linestyle='dashed', color='#696969')
        self.fig.legend()
        labels = self.fig.get_xticklabels() + self.fig.get_yticklabels()
        [label.set_color('#282828') for label in labels]
        [label.set_fontname('Arial') for label in labels]
        [label.set_size(16) for label in labels]
        self.fig.spines['left'].set_color('black')
        self.fig.spines['bottom'].set_color('black')


class CurvesWidget(QWidget):
    def __init__(self):
        super(CurvesWidget, self).__init__()
        self.colorlist = ['#377EB8', '#FF1493', '#984EA3', '#FF7F00', '#A65628', '#F781BF', '#999999', '#4DAF4A',
                          '#D2691E', '#DEB887']
        self.fontdict = {
            'family': 'Arial',
            'size': 16,
            'color': '#282828',
        }
        self.initUI()

    def initUI(self):
        self.setWindowTitle('iLearnPlus Curve')
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.figureCanvas = MyFigureCanvas()
        layout.addWidget(self.figureCanvas)
        self.navigationToolbar = NavigationToolbar2QT(self.figureCanvas, self)
        spacer = QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.comboBox = QComboBox()
        self.comboBox.setFont(QFont('Arial', 8))
        self.comboBox.addItems(['ROC', 'PRC'])
        self.pvalueBtn = QPushButton(' P values ')
        self.pvalueBtn.clicked.connect(self.calculate_pvalue)
        hLayout = QHBoxLayout()
        hLayout.addStretch(1)
        hLayout.addWidget(self.navigationToolbar)
        hLayout.addItem(spacer)
        hLayout.addWidget(self.comboBox)
        hLayout.addWidget(self.pvalueBtn)
        hLayout.addStretch(1)
        layout.addLayout(hLayout)

    def init_roc_data(self, type, title, data):
        self.title = title
        self.data = data
        self.type = type
        if type == 0:
            self.x = 'fpr'
            self.y = 'tpr'
        else:
            self.x = 'recall'
            self.y = 'precision'
        self.fig = self.figureCanvas.figure.add_subplot(121)
        self.__draw_roc_figure__()

    def __draw_roc_figure__(self):
        self.fig.cla()
        self.fig.set_facecolor('white')
        self.fig.set_title(self.title, fontfamily='Arial', fontsize=18)
        self.fig.set_xlabel("False positive rate", fontdict=self.fontdict)
        self.fig.set_ylabel("True positive rate", fontdict=self.fontdict)
        self.fig.set_xticks(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        self.fig.set_yticks(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        for i, item in enumerate(self.data.keys()):
            self.fig.plot(self.data[item][1][self.x], self.data[item][1][self.y], label='%s %s' %(item, self.data[item][0]), color=self.colorlist[i%len(self.colorlist)], lw=2, alpha=1.0)
        if self.type == 0:
            self.fig.plot([0, 1], [0, 1], label='Random', lw=2, alpha=0.5, linestyle='dashed', color='#696969')
        self.fig.legend()
        labels = self.fig.get_xticklabels() + self.fig.get_yticklabels()
        [label.set_color('#282828') for label in labels]
        [label.set_fontname('Arial') for label in labels]
        [label.set_size(16) for label in labels]
        self.fig.spines['left'].set_color('black')
        self.fig.spines['bottom'].set_color('black')

    def init_prc_data(self, type, title, data):
        self.title = title
        self.prc_data = data
        self.prc_type = type
        if self.prc_type == 0:
            self.prc_x = 'fpr'
            self.prc_y = 'tpr'
        else:
            self.prc_x = 'recall'
            self.prc_y = 'precision'
        self.prc = self.figureCanvas.figure.add_subplot(122)
        self.__draw_prc_figure__()

    def __draw_prc_figure__(self):
        self.prc.cla()
        self.prc.set_facecolor('white')
        self.prc.set_title(self.title, fontfamily='Arial', fontsize=18)
        self.prc.set_xlim((-0.05, 1.05))
        self.prc.set_ylim((-0.05, 1.05))
        self.prc.set_xlabel("Recall", fontdict=self.fontdict)
        self.prc.set_ylabel("Precision", fontdict=self.fontdict)
        self.prc.set_xticks(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        self.prc.set_yticks(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        for i, item in enumerate(self.prc_data.keys()):
            self.prc.plot(self.prc_data[item][1][self.prc_x], self.prc_data[item][1][self.prc_y], label='%s %s' %(item, self.prc_data[item][0]), color=self.colorlist[i%len(self.colorlist)], lw=2, alpha=1.0)
        self.prc.legend()
        labels = self.prc.get_xticklabels() + self.prc.get_yticklabels()
        [label.set_color('#282828') for label in labels]
        [label.set_fontname('Arial') for label in labels]
        [label.set_size(16) for label in labels]
        self.prc.spines['left'].set_color('black')
        self.prc.spines['bottom'].set_color('black')

    def init_prediction_scores(self, task, prediction_score):
        self.task = task
        self.prediction_scores = prediction_score

    def calculate_pvalue(self):
        try:
            if self.task == 'binary':
                method, bootstrap_n, ok = InputDialog.QStaticsInput.getValues()
                type = self.comboBox.currentText()
                if ok:
                    self.subWin = BootstrapTestWidget(self.prediction_scores, bootstrap_n, type)
                    self.subWin.setWindowTitle('Calculating p values ... ')
                    t = threading.Thread(target=self.subWin.bootstrapTest)
                    t.start()
                    self.subWin.show()
            else:
                QMessageBox.warning(self, 'Warning', 'Only be used in binary classification task.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)


class CustomCurveWidget(QWidget):
    def __init__(self):
        super(CustomCurveWidget, self).__init__()
        self.colorlist = ['#377EB8', '#FF1493', '#984EA3', '#FF7F00', '#A65628', '#F781BF', '#999999', '#4DAF4A',
                          '#D2691E', '#DEB887']
        self.fontdict = {
            'family': 'Arial',
            'size': 16,
            'color': '#282828',
        }
        self.initUI()

    def initUI(self):
        self.setWindowTitle('iLearnPlus Curve')
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.resize(800, 600)
        layout = QVBoxLayout(self)
        self.figureCanvas = MyFigureCanvas()
        layout.addWidget(self.figureCanvas)
        self.navigationToolbar = NavigationToolbar2QT(self.figureCanvas, self)
        hLayout = QHBoxLayout()
        hLayout.addStretch(1)
        hLayout.addWidget(self.navigationToolbar)
        hLayout.addStretch(1)
        layout.addLayout(hLayout)

    def init_data(self, type, title, fold_data):
        self.title = title
        self.fold_data = fold_data
        self.type = type
        if type == 0:
            self.x = 'fpr'
            self.y = 'tpr'
        else:
            self.x = 'recall'
            self.y = 'precision'
        self.fig = self.figureCanvas.figure.add_subplot(111)
        self.__draw_figure__()

    def __draw_figure__(self):
        self.fig.cla()
        self.fig.set_facecolor('white')
        self.fig.set_xlim((-0.05, 1.05))
        self.fig.set_ylim((-0.05, 1.05))
        self.fig.set_title(self.title)
        if self.title == 'ROC curve':
            x_label = 'False positive rate'
            y_label = 'True positive rate'
        else:
            x_label = 'Recall'
            y_label = 'Precision'
        self.fig.set_xlabel(x_label, fontdict=self.fontdict)
        self.fig.set_ylabel(y_label, fontdict=self.fontdict)
        self.fig.set_xticks(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        self.fig.set_yticks(np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        for i, item in enumerate(self.fold_data):
            self.fig.plot(item[1][self.x], item[1][self.y], label=item[0], lw=item[2], linestyle=item[3], color=item[4], alpha=1.0)
        if self.type == 0:
            self.fig.plot([0, 1], [0, 1], label='Random', lw=2, alpha=0.5, linestyle='dashed', color='#696969')
        self.fig.legend()
        labels = self.fig.get_xticklabels() + self.fig.get_yticklabels()
        [label.set_color('#282828') for label in labels]
        [label.set_fontname('Arial') for label in labels]
        [label.set_size(16) for label in labels]
        self.fig.spines['left'].set_color('black')
        self.fig.spines['bottom'].set_color('black')

