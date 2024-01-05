import pickle

from model import KD_EGNN
from MY_data import myDatasets
import gensim.models
import argparse
from torch.utils.data import Dataset, DataLoader
import sys, os, re
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QFileDialog, QLabel, QHBoxLayout, QGroupBox, QTextEdit,
                             QVBoxLayout, QSplitter, QTableWidget, QTabWidget,
                             QTableWidgetItem, QMessageBox, QFormLayout, QRadioButton,
                             QHeaderView,
                             QAbstractItemView)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, pyqtSignal
from util import PlotWidgets
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tools import evaluate,analysis
from util.EvaluationMetrics import Metrics
import qdarkstyle
from PyQt5 import sip
import joblib
from qdarkstyle.light.palette import LightPalette
from qt_material import apply_stylesheet
from sklearn import metrics
class Args:
	def __init__(self) -> None:
		self.batch_size = 1
		self.lr = 0.0001
		# self.epochs = 60
		self.epochs = 50
		self.numworker = 4
		self.infeature_size = 1280
		self.outfeature_size = 512
		self.nhidden_eg = 128
		self.edge_feature = 0
		self.n_eglayer = 4
		self.nclass = 2
		self.temperature = 3
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.loss_coefficient = 0.3
		self.feature_loss_coefficient = 0.03
		self.model = 'KD_EGNN'
args = Args()
class SEKDLoadModel(QWidget):
    close_signal = pyqtSignal(str)
    def __init__(self):
        super(SEKDLoadModel, self).__init__()

        """ Machine Learning Variable """
        self.data_index = {
            'Training_data': None,
            'Testing_data': None,
            'Training_score': None,
            'Testing_score': None,
            'Metrics': None,
            'ROC': None,
            'PRC': None,
            'Model': None,
        }
        self.current_data_index = 0
        self.ml_running_status = False

        self.model_list = []
        self.dataframe = None
        self.datalabel = None
        self.score = None
        self.metrics = None
        self.aucData = None
        self.prcData = None

        # initialize UI
        self.initUI()

    def initUI(self):
        self.setWindowTitle('CLQ LoadModel')
        self.resize(800, 600)
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowIcon(QIcon('images/logo1.jpg'))

        # file
        topGroupBox = QGroupBox('Load data', self)
        topGroupBox.setFont(QFont('Arial', 10, QFont.Bold))
        topGroupBox.setMinimumHeight(100)
        topGroupBoxLayout = QFormLayout()
        modelFileButton = QPushButton('Load')
        modelFileButton.setToolTip('One or more models could be loaded.')
        modelFileButton.clicked.connect(self.loadModel)
        testFileButton = QPushButton('Open')
        testFileButton.clicked.connect(self.loadFile)

        # yFileButton = QPushButton('Open')
        # yFileButton.clicked.connect(self.loadyFile)
        topGroupBoxLayout.addRow('Open model file(s):', modelFileButton)
        topGroupBoxLayout.addRow('Open testing file:', testFileButton)
        # topGroupBoxLayout.addRow('Open y file:', yFileButton)
        topGroupBox.setLayout(topGroupBoxLayout)

        # start button
        startGroupBox = QGroupBox('Operator', self)
        startGroupBox.setFont(QFont('Arial', 10, QFont.Bold))
        startLayout = QHBoxLayout(startGroupBox)
        self.ml_start_button = QPushButton('Start')
        self.ml_start_button.clicked.connect(self.run_model)
        self.ml_start_button.setFont(QFont('Arial', 10, QFont.Bold))
        self.ml_save_button = QPushButton('Save')
        self.ml_save_button.setFont(QFont('Arial', 10, QFont.Bold))
        # self.ml_save_button.clicked.connect(self.save_ml_files)
        startLayout.addWidget(self.ml_start_button)
        startLayout.addWidget(self.ml_save_button)

        # log
        logGroupBox = QGroupBox('Operation Status', self)
        logGroupBox.setFont(QFont('Arial', 10, QFont.Bold))
        logLayout = QHBoxLayout(logGroupBox)
        self.logTextEdit = QTextEdit()
        self.logTextEdit.setFont(QFont('Arial', 8, QFont.Bold))
        logLayout.addWidget(self.logTextEdit)

        ### layout
        left_vertical_layout = QVBoxLayout()
        left_vertical_layout.addWidget(topGroupBox)
        left_vertical_layout.addWidget(logGroupBox)
        left_vertical_layout.addWidget(startGroupBox)

        #### widget
        leftWidget = QWidget()
        leftWidget.setLayout(left_vertical_layout)

        #### view region
        # scoreTabWidget = QTabWidget()
        # trainScoreWidget = QWidget()
        # scoreTabWidget.setFont(QFont('Arial', 8, QFont.Bold))
        # scoreTabWidget.addTab(trainScoreWidget, 'Prediction score and evaluation metrics')
        # train_score_layout = QVBoxLayout(trainScoreWidget)
        # self.train_score_tableWidget = QTableWidget()
        # self.train_score_tableWidget.setFont(QFont('Arial', 8, QFont.Bold))
        # self.train_score_tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # self.train_score_tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # train_score_layout.addWidget(self.train_score_tableWidget)

        self.metricsTableWidget = QTableWidget()
        self.metricsTableWidget.setFont(QFont('Arial', 8, QFont.Bold))
        self.metricsTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 设置列宽
        self.metricsTableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metricsTableWidget.resizeRowsToContents()
        # splitter_middle = QSplitter(Qt.Vertical)
        # splitter_middle.addWidget(scoreTabWidget)
        # splitter_middle.addWidget(self.metricsTableWidget)

        # self.dataTableWidget = QTableWidget(2, 4)
        # self.dataTableWidget.setFont(QFont('Arial', 8, QFont.Bold))
        # self.dataTableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # self.dataTableWidget.setShowGrid(False)
        # self.dataTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.dataTableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        # self.dataTableWidget.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        # self.dataTableWidget.setHorizontalHeaderLabels(['Select', 'Data', 'Shape', 'Source'])
        # self.dataTableWidget.verticalHeader().setVisible(False)
        data_middle = QSplitter(Qt.Vertical)
        data_middle.addWidget(self.metricsTableWidget)
        # data_middle.addWidget(self.dataTableWidget)

        self.roc_curve_widget = PlotWidgets.CurveWidget()
        self.prc_curve_widget = PlotWidgets.CurveWidget()
        plotTabWidget = QTabWidget()
        plotTabWidget.setFont(QFont('Arial', 8, QFont.Bold))
        rocWidget = QWidget()
        self.rocLayout = QVBoxLayout(rocWidget)
        self.rocLayout.addWidget(self.roc_curve_widget)
        prcWidget = QWidget()
        self.prcLayout = QHBoxLayout(prcWidget)
        self.prcLayout.addWidget(self.prc_curve_widget)
        plotTabWidget.addTab(rocWidget, 'ROC curve')
        plotTabWidget.addTab(prcWidget, 'PRC curve')
        splitter_right = QSplitter(Qt.Vertical)
        splitter_right.addWidget(plotTabWidget)
        splitter_right.addWidget(data_middle)

        splitter_right.setSizes([800, 300])

        splitter_view = QSplitter(Qt.Horizontal)
        # splitter_view.addWidget(scoreTabWidget)
        splitter_view.addWidget(splitter_right)

        splitter_view.setSizes([100, 500])

        ##### splitter
        splitter_1 = QSplitter(Qt.Horizontal)
        splitter_1.addWidget(leftWidget)
        splitter_1.addWidget(splitter_view)
        splitter_1.setSizes([100, 1000])

        ###### vertical layout
        vLayout = QVBoxLayout()

        ## status bar
        statusGroupBox = QGroupBox('Status', self)
        statusGroupBox.setFont(QFont('Arial', 10, QFont.Bold))
        statusLayout = QHBoxLayout(statusGroupBox)
        self.ml_status_label = QLabel('Welcome to SEKD-PPIS')
        self.ml_progress_bar = QLabel()
        self.ml_progress_bar.setMaximumWidth(230)
        statusLayout.addWidget(self.ml_status_label)
        statusLayout.addWidget(self.ml_progress_bar)

        splitter_2 = QSplitter(Qt.Vertical)
        splitter_2.addWidget(splitter_1)
        splitter_2.addWidget(statusGroupBox)
        splitter_2.setSizes([1000, 100])
        vLayout.addWidget(splitter_2)
        self.setLayout(vLayout)

    def loadModel(self):
        # 使用QFileDialog.getOpenFileNames方法打开文件选择对话框，获取用户选择的模型文件列表
        model_files, ok = QFileDialog.getOpenFileNames(self,'models')
        if len(model_files) > 0:
            self.model_list = []
            for file in model_files:
                try:
                    model = KD_EGNN(infeature_size=args.infeature_size, outfeature_size=args.outfeature_size,
                                    nhidden_eg=args.nhidden_eg, edge_feature=args.edge_feature,
                                    n_eglayer=args.n_eglayer,
                                    nclass=args.nclass, device=args.device)
                    if torch.cuda.is_available():
                        model.cuda()

                    models = torch.load(file,map_location='cuda:0')
                    keys = []
                    for key, wight in models.items():
                    # print(key)
                        keys.append(key)
                    newkey = keys[-6:]
                    for key in newkey:
                        del models[key]
                    model.load_state_dict(models)
                    model.eval()
                    self.model_list.append(model)
                except Exception as e:
                    # 如果加载模型失败，弹出错误提示框，并清空模型列表，返回False
                    QMessageBox.critical(self, 'Error', 'Load model failed.', QMessageBox.Ok | QMessageBox.No,
                                         QMessageBox.Ok)
                    self.model_list = []
                    return False
                    # 如果所有模型都加载成功，在日志文本编辑器中添加成功信息和模型数量
            self.logTextEdit.append('Load model successfully.')
            self.logTextEdit.append('Model number: %s' % len(model_files))
            return True
        else:
            return False

    def loadFile(self):
        file, ok = QFileDialog.getOpenFileName(self, 'data')
        if ok:
            if not os.path.exists(file):
                QMessageBox.critical(self, 'Error', 'Data file does not exist.', QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)
                return False
            self.dataframe= None
            try:
                with open(file, "rb") as f:
                    test_data = pickle.load(f)
                IDs, sequences, labels = [], [], []
                for ID in test_data:
                    IDs.append(ID)
                    item = test_data[ID]
                    sequences.append(item[0])
                    labels.append(item[1])
                test_dic = {"ID": IDs, "sequence": sequences, "label": labels}
                test_dataframe = pd.DataFrame(test_dic)
                test_dataset = DataLoader(dataset=myDatasets(test_dataframe), batch_size=1, shuffle=True, num_workers=0)
                self.dataframe = test_dataset
            except Exception as e:
                QMessageBox.critical(self, 'Error', 'Open data file failed.', QMessageBox.Ok | QMessageBox.No,
                                     QMessageBox.Ok)
                return False
            self.logTextEdit.append('Load data file successfully.')
            return True
        else:
            return False

    def run_model(self):
        self.result = None
        self.metrics = None
        if len(self.model_list) > 0 and not self.dataframe is None:
            try:
                for model in self.model_list:
                    epoch_loss_test_avg, test_true, test_pred, pred_dict = evaluate(model, self.dataframe)
                    result_test = analysis(test_true, test_pred)
                column_name = ['ACC', 'PRE', 'REC', 'F1', 'AUC', 'AUPRC', 'MCC','threshold']
                metrics_ind = np.array([result_test['ACC'],result_test['PRE'],result_test['REC'],result_test['F1'],
                                        result_test['AUC'],result_test['AUPRC'],result_test['MCC'],
                                        result_test['threshold']]).reshape((1, -1))
                index_name = ['Metrics value']
                self.aucData = ['AUROC = %s' % result_test['AUC'], result_test['AUCDot']]
                self.prcData = ['AUPRC = %s' % result_test['AUPRC'],result_test['PRCDot']]
                self.metrics = pd.DataFrame(metrics_ind, index=index_name, columns=column_name)

                data = self.metrics.values
                self.metricsTableWidget.setRowCount(data.shape[0])
                self.metricsTableWidget.setColumnCount(data.shape[1])
                self.metricsTableWidget.setHorizontalHeaderLabels(
                    ['ACC', 'PRE', 'REC', 'F1', 'AUC', 'AUPRC', 'MCC','threshold'])
                self.metricsTableWidget.setVerticalHeaderLabels(self.metrics.index)
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        cell = QTableWidgetItem(str(data[i][j]))
                        self.metricsTableWidget.setItem(i, j, cell)
                # if self.data_index['Metrics'] is None:
                #     # index = self.current_data_index
                #     index = 1
                #     self.data_index['Metrics'] = index
                #     self.dataTableWidget.insertRow(index)
                #     self.current_data_index += 1
                # else:
                    # index = self.data_index['Metrics']
                    index = 1
                # self.metrics_radio = QRadioButton()
                # self.dataTableWidget.setCellWidget(index, 0, self.metrics_radio)
                # self.dataTableWidget.setItem(index, 1, QTableWidgetItem('Evaluation metrics'))
                # self.dataTableWidget.setItem(index, 2, QTableWidgetItem(str(data.shape)))
                # self.dataTableWidget.setItem(index, 3, QTableWidgetItem('NA'))

                # plot ROC
                if not self.aucData is None:
                    self.rocLayout.removeWidget(self.roc_curve_widget)
                    sip.delete(self.roc_curve_widget)
                    self.roc_curve_widget = PlotWidgets.CurveWidget()
                    self.roc_curve_widget.init_data(0, 'ROC curve', ind_data=self.aucData)
                    self.rocLayout.addWidget(self.roc_curve_widget)

                    # plot PRC
                if not self.prcData is None:
                    self.prcLayout.removeWidget(self.prc_curve_widget)
                    sip.delete(self.prc_curve_widget)
                    self.prc_curve_widget = PlotWidgets.CurveWidget()
                    self.prc_curve_widget.init_data(1, 'PRC curve', ind_data=self.prcData)
                    self.prcLayout.addWidget(self.prc_curve_widget)
            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e), QMessageBox.Ok | QMessageBox.No, QMessageBox.Ok)

        else:
            QMessageBox.critical(self, 'Error', 'Please load the model file(s) or data file.', QMessageBox.Ok | QMessageBox.No,
                         QMessageBox.Ok)










if __name__ == '__main__':
    app = QApplication(sys.argv)

    extra = {

        # Density Scale
        'density_scale': '2',
    }
    apply_stylesheet(app, theme='dark_cyan.xml',extra=extra)
    # window.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', palette=LightPalette()))
    window = SEKDLoadModel()
    window.show()
    sys.exit(app.exec_())