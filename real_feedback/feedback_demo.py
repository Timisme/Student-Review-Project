from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
import numpy as np 
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from info_extraction import info_extractor
jieba.set_dictionary('jieba_traditional.txt')

df_clean = pd.read_csv('df_clean.csv')

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1108, 786)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.show_sent_plots = QtWidgets.QPushButton(self.centralwidget)
        self.show_sent_plots.setGeometry(QtCore.QRect(60, 580, 431, 81))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.show_sent_plots.setFont(font)
        self.show_sent_plots.setObjectName("show_sent_plots")

        self.keyword_input = QtWidgets.QLineEdit(self.centralwidget)
        self.keyword_input.setGeometry(QtCore.QRect(650, 160, 391, 81))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.keyword_input.setFont(font)
        self.keyword_input.setObjectName("keyword_input")

        self.file_input = QtWidgets.QLineEdit(self.centralwidget)
        self.file_input.setGeometry(QtCore.QRect(70, 140, 481, 81))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.file_input.setFont(font)
        self.file_input.setObjectName("file_input")

        self.model_test = QtWidgets.QPushButton(self.centralwidget)
        self.model_test.setGeometry(QtCore.QRect(70, 350, 291, 91))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.model_test.setFont(font)
        self.model_test.setObjectName("model_test")

        self.txt_keyword_query = QtWidgets.QPushButton(self.centralwidget)
        self.txt_keyword_query.setGeometry(QtCore.QRect(650, 640, 311, 81))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.txt_keyword_query.setFont(font)
        self.txt_keyword_query.setObjectName("txt_keyword_query")

        self.txt_keyword_query.clicked.connect(self.show_keywords_pressed)

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(60, 60, 381, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(60, 270, 381, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(60, 510, 401, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(630, 50, 461, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")

        self.keyword_extract = QtWidgets.QPushButton(self.centralwidget)
        self.keyword_extract.setGeometry(QtCore.QRect(650, 340, 331, 81))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.keyword_extract.setFont(font)
        self.keyword_extract.setObjectName("keyword_extract")

        self.keyword_extract.clicked.connect(self.show_related_txt_pressed)

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(650, 100, 461, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(640, 490, 461, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")

        self.n_txt = QtWidgets.QLineEdit(self.centralwidget)
        self.n_txt.setGeometry(QtCore.QRect(650, 260, 251, 61))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.n_txt.setFont(font)
        self.n_txt.setObjectName("n_txt")

        self.index_input = QtWidgets.QLineEdit(self.centralwidget)
        self.index_input.setGeometry(QtCore.QRect(650, 550, 221, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.index_input.setFont(font)
        self.index_input.setObjectName("index_input")

        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(900, 550, 131, 51))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1108, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.show_sent_plots.setText(_translate("MainWindow", "學生回饋情緒總結圖表"))
        self.keyword_input.setText(_translate("MainWindow", "輸入查詢關鍵字"))
        self.file_input.setText(_translate("MainWindow", "輸入學生回饋檔案名稱(.csv)"))
        self.model_test.setText(_translate("MainWindow", "情緒分析"))
        self.txt_keyword_query.setText(_translate("MainWindow", "查詢回饋關鍵字"))
        self.label.setText(_translate("MainWindow", "1. 請輸入學生回饋檔案名稱"))
        self.label_2.setText(_translate("MainWindow", "2. 點擊按鈕進行模型分析"))
        self.label_3.setText(_translate("MainWindow", "3. 產生整體學生回饋情緒圖表"))
        self.label_4.setText(_translate("MainWindow", "4. 輸入關鍵字或句子"))
        self.keyword_extract.setText(_translate("MainWindow", "查詢學生回饋文本"))
        self.label_5.setText(_translate("MainWindow", "找出最符合回饋文本"))
        self.label_6.setText(_translate("MainWindow", "5. 學生回饋關鍵字查詢"))
        self.n_txt.setText(_translate("MainWindow", "輸入查詢回饋數量(正整數)"))
        self.index_input.setText(_translate("MainWindow", "輸入回饋編號"))
        self.comboBox.setItemText(0, _translate("MainWindow", "正面回饋"))
        self.comboBox.setItemText(1, _translate("MainWindow", "負面回饋"))

    def show_related_txt_pressed(self):

        txt = self.keyword_input.text()
        k = int(self.n_txt.text())

        info_extractor(df_clean).document_matching(txt, k)

    def show_keywords_pressed(self):

        idx = int(self.index_input.text())

        if self.comboBox.currentText() == '正面回饋':
            pos = True
        else: 
            pos = False
            # print(self.comboBox.currentText())
        
        info_extractor(df_clean).keyword_extraction(idx= idx, pos=pos)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
