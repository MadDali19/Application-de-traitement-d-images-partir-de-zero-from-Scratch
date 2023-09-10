import os
import math
import numpy as np
from PyQt5 import QtWidgets,QtCore,QtGui
from cv2 import cv2
import imutils, sys
from PIL import Image
from matplotlib import pyplot as plt
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
h=np.arange(256)
m1=3
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(694, 394)
        MainWindow.setStyleSheet(u"background-color: rgb(0, 0, 0);")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.splitter_2 = QSplitter(self.centralwidget)
        self.splitter_2.setObjectName(u"splitter_2")
        self.splitter_2.setOrientation(Qt.Horizontal)
        self.layoutWidget = QWidget(self.splitter_2)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.verticalLayout_2 = QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.Hist = QPushButton(self.layoutWidget)
        self.Hist.setObjectName(u"Hist")
        self.Hist.setStyleSheet(u"#Hist{background-color: rgb(255, 255, 255);}\n"
                                "#Hist:hover{background-color: rgb(255, 11, 3);}\n"
                                "\n"
                                "\n"
                                "")

        self.horizontalLayout_12.addWidget(self.Hist)

        self.Histcouleur = QPushButton(self.layoutWidget)
        self.Histcouleur.setObjectName(u"Histcouleur")
        self.Histcouleur.setStyleSheet(u"#Histcouleur{background-color: rgb(255, 255, 255);}\n"
                                       "#Histcouleur:hover{background-color: rgb(255, 11, 3);}\n"
                                       "\n"
                                       "\n"
                                       "")

        self.horizontalLayout_12.addWidget(self.Histcouleur)

        self.verticalLayout_2.addLayout(self.horizontalLayout_12)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.Inversion = QPushButton(self.layoutWidget)
        self.Inversion.setObjectName(u"Inversion")
        self.Inversion.setStyleSheet(u"#Inversion{background-color: rgb(255, 255, 255);}\n"
                                     "#Inversion:hover{background-color: rgb(255, 11, 3);}\n"
                                     "\n"
                                     "\n"
                                     "")

        self.horizontalLayout_4.addWidget(self.Inversion)

        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.Translation = QPushButton(self.layoutWidget)
        self.Translation.setObjectName(u"Translation")
        self.Translation.setStyleSheet(u"#Translation{background-color: rgb(255, 255, 255);}\n"
                                       "#Translation:hover{background-color: rgb(255, 11, 3);}\n"
                                       "\n"
                                       "\n"
                                       "")

        self.horizontalLayout_6.addWidget(self.Translation)

        self.Translation_2 = QPushButton(self.layoutWidget)
        self.Translation_2.setObjectName(u"Translation_2")
        self.Translation_2.setStyleSheet(u"#Translation_2{background-color: rgb(255, 255, 255);}\n"
                                         "#Translation_2:hover{background-color: rgb(255, 11, 3);}\n"
                                         "\n"
                                         "\n"
                                         "")

        self.horizontalLayout_6.addWidget(self.Translation_2)

        self.verticalLayout_2.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.Expansion = QPushButton(self.layoutWidget)
        self.Expansion.setObjectName(u"Expansion")
        self.Expansion.setStyleSheet(u"#Expansion{background-color: rgb(255, 255, 255);}\n"
                                     "#Expansion:hover{background-color: rgb(255, 11, 3);}\n"
                                     "\n"
                                     "\n"
                                     "")

        self.horizontalLayout_7.addWidget(self.Expansion)

        self.Egalisation = QPushButton(self.layoutWidget)
        self.Egalisation.setObjectName(u"Egalisation")
        self.Egalisation.setStyleSheet(u"#Egalisation{background-color: rgb(255, 255, 255);}\n"
                                       "#Egalisation:hover{background-color: rgb(255, 11, 3);}\n"
                                       "\n"
                                       "\n"
                                       "")

        self.horizontalLayout_7.addWidget(self.Egalisation)

        self.verticalLayout_2.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.SContour = QPushButton(self.layoutWidget)
        self.SContour.setObjectName(u"SContour")
        self.SContour.setStyleSheet(u"#SContour{background-color: rgb(255, 255, 255);}\n"
                                    "#SContour:hover{background-color: rgb(255, 11, 3);}\n"
                                    "\n"
                                    "\n"
                                    "")

        self.horizontalLayout_8.addWidget(self.SContour)

        self.Binaristion = QPushButton(self.layoutWidget)
        self.Binaristion.setObjectName(u"Binaristion")
        self.Binaristion.setStyleSheet(u"#Binaristion{background-color: rgb(255, 255, 255);}\n"
                                       "#Binaristion:hover{background-color: rgb(255, 11, 3);}")

        self.horizontalLayout_8.addWidget(self.Binaristion)

        self.verticalLayout_2.addLayout(self.horizontalLayout_8)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.Etiquetage = QPushButton(self.layoutWidget)
        self.Etiquetage.setObjectName(u"Etiquetage")
        self.Etiquetage.setStyleSheet(u"#Etiquetage{background-color: rgb(255, 255, 255);}\n"
                                      "#Etiquetage:hover{background-color: rgb(255, 11, 3);}")

        self.horizontalLayout_9.addWidget(self.Etiquetage)

        self.SRegion = QPushButton(self.layoutWidget)
        self.SRegion.setObjectName(u"SRegion")
        self.SRegion.setStyleSheet(u"#SRegion{background-color: rgb(255, 255, 255);}\n"
                                   "#SRegion:hover{background-color: rgb(255, 11, 3);}")

        self.horizontalLayout_9.addWidget(self.SRegion)

        self.verticalLayout_2.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.ExtraVideo = QPushButton(self.layoutWidget)
        self.ExtraVideo.setObjectName(u"ExtraVideo")
        self.ExtraVideo.setStyleSheet(u"#ExtraVideo{background-color: rgb(255, 255, 255);}\n"
                                      "#ExtraVideo:hover{background-color: rgb(255, 11, 3);}")

        self.horizontalLayout_10.addWidget(self.ExtraVideo)

        self.ZoomP = QPushButton(self.layoutWidget)
        self.ZoomP.setObjectName(u"ZoomP")
        self.ZoomP.setStyleSheet(u"#ZoomP{background-color: rgb(255, 255, 255);}\n"
                                 "#ZoomP:hover{background-color: rgb(255, 11, 3);}")

        self.horizontalLayout_10.addWidget(self.ZoomP)

        self.verticalLayout_2.addLayout(self.horizontalLayout_10)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.ZoomM = QPushButton(self.layoutWidget)
        self.ZoomM.setObjectName(u"ZoomM")
        self.ZoomM.setStyleSheet(u"#ZoomM{background-color: rgb(255, 255, 255);}\n"
                                 "#ZoomM:hover{background-color: rgb(255, 11, 3);}")

        self.horizontalLayout_11.addWidget(self.ZoomM)

        self.verticalLayout_2.addLayout(self.horizontalLayout_11)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.HistN = QPushButton(self.layoutWidget)
        self.HistN.setObjectName(u"HistN")
        self.HistN.setStyleSheet(u"#HistN{background-color: rgb(255, 255, 255);}\n"
                                 "#HistN:hover{background-color: rgb(255, 11, 3);}")

        self.horizontalLayout_13.addWidget(self.HistN)

        self.HistC = QPushButton(self.layoutWidget)
        self.HistC.setObjectName(u"HistC")
        self.HistC.setStyleSheet(u"#HistC{background-color: rgb(255, 255, 255);}\n"
                                 "#HistC:hover{background-color: rgb(255, 11, 3);}")

        self.horizontalLayout_13.addWidget(self.HistC)

        self.verticalLayout_2.addLayout(self.horizontalLayout_13)

        self.splitter_2.addWidget(self.layoutWidget)
        self.layoutWidget1 = QWidget(self.splitter_2)
        self.layoutWidget1.setObjectName(u"layoutWidget1")
        self.verticalLayout = QVBoxLayout(self.layoutWidget1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.HistNC = QPushButton(self.layoutWidget1)
        self.HistNC.setObjectName(u"HistNC")
        self.HistNC.setStyleSheet(u"#HistNC{background-color: rgb(255, 255, 255);}\n"
                                  "#HistNC:hover{background-color: rgb(255, 11, 3);}\n"
                                  "\n"
                                  "\n"
                                  "")

        self.horizontalLayout.addWidget(self.HistNC)

        self.pushButton_4 = QPushButton(self.layoutWidget1)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setStyleSheet(u"#pushButton_4{background-color: rgb(255, 255, 255);}\n"
                                        "#pushButton_4:hover{background-color: rgb(255, 11, 3);}\n"
                                        "\n"
                                        "\n"
                                        "")

        self.horizontalLayout.addWidget(self.pushButton_4)

        self.pushButton_5 = QPushButton(self.layoutWidget1)
        self.pushButton_5.setObjectName(u"pushButton_5")
        self.pushButton_5.setStyleSheet(u"#pushButton_5{background-color: rgb(255, 255, 255);}\n"
                                        "#pushButton_5:hover{background-color: rgb(255, 11, 3);}\n"
                                        "\n"
                                        "\n"
                                        "")

        self.horizontalLayout.addWidget(self.pushButton_5)

        self.pushButton_6 = QPushButton(self.layoutWidget1)
        self.pushButton_6.setObjectName(u"pushButton_6")
        self.pushButton_6.setStyleSheet(u"#pushButton_6{background-color: rgb(255, 255, 255);}\n"
                                        "#pushButton_6:hover{background-color: rgb(255, 11, 3);}\n"
                                        "\n"
                                        "\n"
                                        "")

        self.horizontalLayout.addWidget(self.pushButton_6)

        self.importer = QPushButton(self.layoutWidget1)
        self.importer.setObjectName(u"importer")
        self.importer.setStyleSheet(u"#importer{background-color: rgb(255, 255, 255);}\n"
                                    "#importer:hover{background-color: rgb(255, 11, 3);}\n"
                                    "\n"
                                    "\n"
                                    "")

        self.horizontalLayout.addWidget(self.importer)

        self.sauvegarder = QPushButton(self.layoutWidget1)
        self.sauvegarder.setObjectName(u"sauvegarder")
        self.sauvegarder.setStyleSheet(u"#sauvegarder{background-color: rgb(255, 255, 255);}\n"
                                       "#sauvegarder:hover{background-color: rgb(255, 11, 3);}\n"
                                       "\n"
                                       "\n"
                                       "")

        self.horizontalLayout.addWidget(self.sauvegarder)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.splitter = QSplitter(self.layoutWidget1)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Vertical)
        self.espaceimage = QLabel(self.splitter)
        self.espaceimage.setObjectName(u"espaceimage")
        self.espaceimage.setAlignment(Qt.AlignCenter)
        self.splitter.addWidget(self.espaceimage)
        self.taille = QLabel(self.splitter)
        self.taille.setObjectName(u"taille")
        self.taille.setStyleSheet(u"color: rgb(255, 255, 255);")
        self.splitter.addWidget(self.taille)

        self.verticalLayout.addWidget(self.splitter)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.pushButton_7 = QPushButton(self.layoutWidget1)
        self.pushButton_7.setObjectName(u"pushButton_7")
        self.pushButton_7.setStyleSheet(u"#pushButton_7{background-color: rgb(255, 255, 255);}\n"
                                        "#pushButton_7:hover{background-color: rgb(255, 11, 3);}\n"
                                        "\n"
                                        "\n"
                                        "")

        self.horizontalLayout_5.addWidget(self.pushButton_7)

        self.pushButton_8 = QPushButton(self.layoutWidget1)
        self.pushButton_8.setObjectName(u"pushButton_8")
        self.pushButton_8.setStyleSheet(u"#pushButton_8{background-color: rgb(255, 255, 255);}\n"
                                        "#pushButton_8:hover{background-color: rgb(255, 11, 3);}\n"
                                        "\n"
                                        "\n"
                                        "")

        self.horizontalLayout_5.addWidget(self.pushButton_8)

        self.pushButton_9 = QPushButton(self.layoutWidget1)
        self.pushButton_9.setObjectName(u"pushButton_9")
        self.pushButton_9.setStyleSheet(u"#pushButton_9{background-color: rgb(255, 255, 255);}\n"
                                        "#pushButton_9:hover{background-color: rgb(255, 11, 3);}\n"
                                        "\n"
                                        "\n"
                                        "")

        self.horizontalLayout_5.addWidget(self.pushButton_9)

        self.horizontalLayout_2.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.pushButton = QPushButton(self.layoutWidget1)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setStyleSheet(u"#pushButton{background-color: rgb(255, 255, 255);}\n"
                                      "#pushButton:hover{background-color: rgb(255, 11, 3);}\n"
                                      "\n"
                                      "\n"
                                      "")

        self.horizontalLayout_3.addWidget(self.pushButton)

        self.pushButton_2 = QPushButton(self.layoutWidget1)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setStyleSheet(u"#pushButton_2{background-color: rgb(255, 255, 255);}\n"
                                        "#pushButton_2:hover{background-color: rgb(255, 11, 3);}\n"
                                        "\n"
                                        "\n"
                                        "")

        self.horizontalLayout_3.addWidget(self.pushButton_2)

        self.pushButton_3 = QPushButton(self.layoutWidget1)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setStyleSheet(u"#pushButton_3{background-color: rgb(255, 255, 255);}\n"
                                        "#pushButton_3:hover{background-color: rgb(255, 11, 3);}\n"
                                        "\n"
                                        "\n"
                                        "")

        self.horizontalLayout_3.addWidget(self.pushButton_3)

        self.horizontalLayout_2.addLayout(self.horizontalLayout_3)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.splitter_2.addWidget(self.layoutWidget1)

        self.gridLayout.addWidget(self.splitter_2, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 694, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.importer.clicked.connect(self.loadImage)
        self.Hist.clicked.connect(self.vishistogrammesimple)
        self.HistN.clicked.connect(self.vishistNormalise)
        self.HistC.clicked.connect(self.vishistCumule)
        self.HistNC.clicked.connect(self.vishistCumuleNormalise)
        self.Translation.clicked.connect(self.translation)
        self.Inversion.clicked.connect(self.inversion)
        self.Binaristion.clicked.connect(self.binarisation)
        self.Expansion.clicked.connect(self.Expansion_de_dynamique)
        self.Egalisation.clicked.connect(self.egalisation)
        self.pushButton_7.clicked.connect(self.median3)
        self.pushButton_8.clicked.connect(self.median5)
        self.pushButton_9.clicked.connect(self.median9)
        self.pushButton.clicked.connect(self.filterGauss3)
        self.pushButton_2.clicked.connect(self.filterGauss5)
        self.pushButton_3.clicked.connect(self.filterGauss9)
        self.pushButton_4.clicked.connect(self.filterMOY3)
        self.pushButton_5.clicked.connect(self.filterMOY5)
        self.pushButton_6.clicked.connect(self.filterMOY9)
        self.SContour.clicked.connect(self.contour)
        self.ExtraVideo.clicked.connect(self.video)
        self.Histcouleur.clicked.connect(self.vishistcolor)
        #self.Translation_3.clicked.connect(self.binOtsu)
        self.Etiquetage.clicked.connect(self.segR)
        self.SRegion.clicked.connect(self.otsuV )
        self.sauvegarder.clicked.connect(self.savePhoto)
        self.ZoomM.clicked.connect(self.rotationImage)
        self.ZoomP.clicked.connect(self.flipImage)
        self.Translation_2.clicked.connect(self.retoure)
        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi
    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.Hist.setText(QCoreApplication.translate("MainWindow", u"Hist", None))
        self.Histcouleur.setText(QCoreApplication.translate("MainWindow", u"Histcouleur", None))
        self.Inversion.setText(QCoreApplication.translate("MainWindow", u"Inversion", None))
        self.Translation.setText(QCoreApplication.translate("MainWindow", u"Translation +", None))
        self.Translation_2.setText(QCoreApplication.translate("MainWindow", u"Annuler", None))
        self.Expansion.setText(QCoreApplication.translate("MainWindow", u"Expansion", None))
        self.Egalisation.setText(QCoreApplication.translate("MainWindow", u"Egalisation", None))
        self.SContour.setText(QCoreApplication.translate("MainWindow", u"SContour", None))
        self.Binaristion.setText(QCoreApplication.translate("MainWindow", u"Binaristion", None))
        self.Etiquetage.setText(QCoreApplication.translate("MainWindow", u"Etiquetage", None))
        self.SRegion.setText(QCoreApplication.translate("MainWindow", u"Otsu", None))
        self.ExtraVideo.setText(QCoreApplication.translate("MainWindow", u"ExtraVideo", None))
        self.ZoomP.setText(QCoreApplication.translate("MainWindow", u"Flip", None))
        self.ZoomM.setText(QCoreApplication.translate("MainWindow", u"Rotation", None))
        self.HistN.setText(QCoreApplication.translate("MainWindow", u"HistN", None))
        self.HistC.setText(QCoreApplication.translate("MainWindow", u"HistC", None))
        self.HistNC.setText(QCoreApplication.translate("MainWindow", u"HistNC", None))
        self.pushButton_4.setText(QCoreApplication.translate("MainWindow", u"moyenneur 3x3", None))
        self.pushButton_5.setText(QCoreApplication.translate("MainWindow", u"moyenneur 5x5", None))
        self.pushButton_6.setText(QCoreApplication.translate("MainWindow", u"moyenneur 9x9", None))
        self.importer.setText(QCoreApplication.translate("MainWindow", u"importer", None))
        self.sauvegarder.setText(QCoreApplication.translate("MainWindow", u"sauvegarder", None))
        self.espaceimage.setText("")
        self.taille.setText("")
        self.pushButton_7.setText(QCoreApplication.translate("MainWindow", u"mediane 3x3", None))
        self.pushButton_8.setText(QCoreApplication.translate("MainWindow", u"mediane 5x5", None))
        self.pushButton_9.setText(QCoreApplication.translate("MainWindow", u"mediane 9x9", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"Gauss 3x3", None))
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u"Gauss 5x5", None))
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u"Gauss 9x9", None))
        self.filename = None
        self.tmp = None
        self.IMG=None
        self.filename=None
    # retranslateUi
    def loadImage(self):
            """ Cette fonction chargera l'image sélectionnée par l'utilisateur et
            la définira comme label dans l'interface à l'aide de la fonction setPhoto
            """
            self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
            if (self.filename):
                    self.image = cv2.imread(self.filename,cv2.IMREAD_GRAYSCALE)
                    #self.image = cv2.resize(self.image, [200,200])
                    self.IMG=cv2.imread(self.filename,cv2.IMREAD_GRAYSCALE)
                    #self.IMG=cv2.resize(self.IMG, [200,200])
                    self.width, self.height = Image.open(self.filename).size
                    self.width = str(self.width)
                    self.height = str(self.height)
                    bilgi = "(" + self.width + "x" + self.height + ")"
                    self.taille.setText(bilgi)
                    self.setPhoto(self.image)
    def setPhoto(self, image):
            """ Cette fonction prendra l'entrée d'image et la redimensionnera uniquement
            à des fins d'affichage et la convertira en QImage pour la définir sur le label.
            """
            self.IMG = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
            #self.IMG = cv2.resize(self.IMG, [200, 200])
            self.tmp = image
            image = imutils.resize(image, height=800, width=800)
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.espaceimage.setPixmap(QtGui.QPixmap.fromImage(image))
    def HistS(self):
        """
        histogramme simple
        """
        H = np.zeros(256, int)
        n = len(self.tmp)
        m = len(self.tmp[0])
        for i in range(n):
            for j in range(m):
                H[self.tmp[i][j]] += 1
        return H
    def vishistogrammesimple(self):
        """visualisation"""
        H=self.HistS()
        plt.bar(h, H, 0.8, edgecolor='blue')
        plt.ylabel("nombre d'occurence")
        plt.xlabel("pixels")
        plt.title("Histogramme simple")
        plt.show()
    def Normalise(self):
        """histogramme normalise"""
        H=self.HistS()
        n = len(self.tmp)
        m = len(self.tmp[0])
        nm = n * m
        C = []
        for i in range(len(H)):
            C.append((H[i] / nm).__round__(2))
        return C
    def vishistNormalise(self):
        """visualisation"""
        C=self.Normalise()
        plt.bar(h, C, 0.8, edgecolor='blue')
        plt.ylabel("nombre d'occurence")
        plt.xlabel("pixels")
        plt.title("Histogramme Normalisé")
        plt.show()
    def histCumule(self):
        """histogramme cummule"""
        H = self.HistS()
        C = []
        C.append(H[0])
        for i in range(1, len(H)):
            C.append(C[i - 1] + H[i])
        return C
    def vishistCumule(self):
        """visualisation"""
        C = self.histCumule()
        plt.bar(h, C, 0.8, edgecolor='blue')
        plt.ylabel("nombre d'occurence")
        plt.xlabel("pixels")
        plt.title("Histogramme Cumulé")
        plt.show()
    def histCumuleNormalise(self):
        """histogramme cummule normalise"""
        H = self.Normalise()
        C = []
        C.append(H[0])
        for i in range(1, len(H)):
            C.append(C[i - 1] + H[i])
        return C
    def vishistCumuleNormalise(self):
        """visualisation"""
        C = self.histCumuleNormalise()
        plt.bar(h, C, 0.8, edgecolor='blue')
        plt.ylabel("nombre d'occurence")
        plt.xlabel("pixels")
        plt.title("Histogramme Cumulé Normalisé")
        plt.show()
    def translation(self):
        """translation (claire)"""
        n = len(self.tmp)
        m = len(self.tmp[0])
        Max = self.tmp.max()
        nb = 255- Max
        for i in range(n):
            for j in range(m):
                cm=self.tmp[i][j] + nb
                self.tmp[i][j] = cm
        self.setPhoto(self.tmp)
    def inversion(self):
        """inversion"""
        n = len(self.tmp)
        m = len(self.tmp[0])
        for i in range(n):
            for j in range(m):
                self.tmp[i][j] = 255 - self.tmp[i][j]
        self.setPhoto(self.tmp)
    def binarisation(self):
        """binarisation"""
        for i in range(len(self.tmp)):
            for j in range(len((self.tmp[0]))):
                if self.tmp[i][j] > 180:
                    self.tmp[i][j] = 255
                else:
                    self.tmp[i][j] = 0
        self.setPhoto(self.tmp)
    def Expansion_de_dynamique(self):
        """Expansion de dynamique"""
        n = len(self.tmp)
        m = len(self.tmp[0])
        Max= self.tmp.max()
        Min=self.tmp.min()
        for i in range(n):
            for j in range(m):
                self.tmp[i][j] = (255 / (Max - Min)) * (self.tmp[i][j] - Min)
        self.setPhoto(self.tmp)
    def egalisation(self):
        """egalisation"""
        HC = self.histCumuleNormalise()
        n = len(self.tmp)
        m = len(self.tmp[0])
        for i in range(n):
            for j in range(m):
                NiveauInitial = self.tmp[i][j]
                self.tmp[i][j] = 255 * HC[NiveauInitial]  # 255!!! max
        self.setPhoto(self.tmp)
    def median(self ,masque):
        """filtre median"""
        Height = len(self.tmp)
        Width = len(self.tmp[0])
        l = []
        v = int(masque / 2)
        for i in range(Height):
            for j in range(Width):
                for h in range(masque):
                    if ((i + h - v < 0) or (i + h - v  > Height - 1)):
                        l.append(0)
                    else:
                            for k in range(masque):
                                if ((j + k - v < 0) or (j + k - v > Width - 1)):
                                    l.append(0)
                                else:l.append(self.tmp[i + h - v][j + k - v])
                l.sort()
                self.tmp[i][j] = l[int(len(l) / 2)]
                l = []
        self.setPhoto(self.tmp)
    def median3(self):
        self.median(3)
    def median5(self):
        self.median(5)
    def median9(self):
        self.median(9)
    def gaussian_filter(self,P, sigma):
        """le masque de gauss"""
        P = int(P / 2)
        coef = 1 / (2 * math.pi * (sigma * sigma))
        h = np.zeros((2 * P + 1, 2 * P + 1))
        som = 0
        for m in range(-P, P + 1):
            for n in range(-P, P + 1):
                h[m + P][n + P] = math.exp(-(n * n + m * m) / (2 * sigma * sigma)) * coef
                som += h[m + P][n + P]
        h = h / som
        return h
    def Gauss(self,kernel, image,p1):
        i_width, i_height = image.shape[0], image.shape[1]
        k_width, k_height = kernel.shape[0], kernel.shape[1]

        filtered = np.zeros_like(image)
        for y in range(i_height):
            for x in range(i_width):
                weighted_pixel_sum = 0
                # De cette façon, le pixel à image[y,x] est multiplié par le noyau[0,0] ; analogue,
                # image[y-1,x] est multiplié par kernel[-1,0] etc.
                for ky in range(-(k_height // p1), k_height - 1):
                    for kx in range(-(k_width // p1), k_width - 1):
                        pixel = 0
                        pixel_y = y - ky
                        pixel_x = x - kx

                        # vérification des limites : toutes les valeurs en dehors de l'image sont traitées comme nulles.
                        if (pixel_y >= 0) and (pixel_y < i_height) and (pixel_x >= 0) and (pixel_x < i_width):
                            pixel = image[pixel_x, pixel_y]
                        weight = kernel[ky + (k_height // p1), kx + (k_width // p1)]
                        weighted_pixel_sum += pixel * weight
                filtered[x, y] = weighted_pixel_sum

        return filtered
    def filterGauss3(self):
        K=self.gaussian_filter(3, 2.8)
        self.tmp=self.Gauss(K,self.tmp,2)
        self.setPhoto(self.tmp)
    def filterGauss5(self):
        K=self.gaussian_filter(5, 2.8)
        self.tmp=self.Gauss(K,self.tmp,4)
        self.setPhoto(self.tmp)
    def filterGauss9(self):
        K=self.gaussian_filter(9, 2.8)
        self.tmp=self.Gauss(K,self.tmp,8)
        self.setPhoto(self.tmp)
    def filterMOY3(self):
        K=np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])
        self.tmp=self.Gauss(K,self.tmp,2)
        self.setPhoto(self.tmp)
    def filterMOY5(self):
        K=np.array([[1/25,1/25,1/25,1/25,1/25],
           [1/25,1/25,1/25,1/25,1/25],
           [1/25,1/25,1/25,1/25,1/25],
           [1/25,1/25,1/25,1/25,1/25],
           [1/25,1/25,1/25,1/25,1/25]])
        self.tmp=self.Gauss(K,self.tmp,4)
        self.setPhoto(self.tmp)
    def conv(self,img, masque, mm):
        """produit convolution"""
        Height = len(img)
        Width = len(img[0])
        s = []
        l = -int(masque / 2) + 1
        k = -int(masque / 2) + 1
        for i in range(Height):
            for j in range(Width):
                for h in range(masque):
                    l += 1
                    ss = []
                    for c in range(masque):
                        k += 1
                        if l + i < 0 or k + j < 0:
                            ss.append(0)
                        elif l + i >= Height or k + j >= Width:
                            ss.append(0)
                        else:
                            ss.append(img[i + l][j + k])
                    k = -int(masque / 2) + 1
                    s.append(ss)
                img[i][j] = np.sum(np.multiply(s, mm))
                s = []
                l = -int(masque / 2) + 1
        return img
    def filterMOY9(self):
        K=np.array([[1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],[1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],[1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],
           [1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],[1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],[1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],
           [1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],[1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81],[1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81,1/81]
           ])
        self.tmp=self.Gauss(K,self.tmp,8)
        self.setPhoto(self.tmp)
    #show
    def contour(self):
        img=np.copy(self.tmp).astype(float) / 255.0
        masqueH=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        H=self.conv(img,3,masqueH)
        img = np.copy(self.tmp).astype(float) / 255.0
        masqueV=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        V=self.conv(img,3,masqueV)
        image=np.zeros((len(self.tmp),len(self.tmp[0])))
        for i in range(len(self.tmp)):
            for j in range(len((self.tmp[0]))):
                image[i][j]=math.sqrt(math.pow(H[i][j],2)+math.pow(V[i][j],2))
        image = cv2.resize(image, (800, 800))
        cv2.imshow('image', image)
        cv2.waitKey(0)
        windows = cv2.destroyAllWindows()
    def histocolor(self,I):
        R = np.zeros(256)
        G = np.zeros(256)
        B = np.zeros(256)
        li=len(I)
        col = len(I[0])
        for i in range(0, li):
            for j in range(0, col):
                r = (I[i][j])[0]
                g = (I[i][j])[1]
                b = (I[i][j])[2]
                R[r] = R[r] + 1
                G[g] = G[g] + 1
                B[b] = B[b] + 1
                # (R,G,B)=histocolor(img)
        return R, G, B
    def vishistcolor(self):
        image=None
        filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        image = cv2.imread(filename)
        (R, G, B) = self.histocolor(image)
        fig, axs = plt.subplots(3)
        axs[0].plot(R, "r")
        axs[1].plot(G, "g")
        axs[2].plot(B, "b")
        plt.show()
    def otsu(self,gray):
            #le nombre de pixel quel'image contient la largueurxla heuteur
            pixel_number = gray.shape[0] * gray.shape[1]
            #le poid de l'image
            mean_weigth = 1.0 / pixel_number
            #l'histograme de l'image la valeuret la repetition
            his, bins = np.histogram(gray, np.array(range(0, 256)))
            final_thresh = -1
            final_value = -1
            for t in bins[1:-1]:
                #le background
                #la somme de l'histogramme diviser par la somme de pixel de l'image
                Wb = np.sum(his[:t]) * mean_weigth
                Wf = np.sum(his[t:]) * mean_weigth
                #la moyenne de l'histogramme
                mub = np.mean(his[:t])
                muf = np.mean(his[t:])
                #s
                value = Wb * Wf * (mub - muf) ** 2

                if value > final_value:
                    final_thresh = t
                    final_value = value
            final_img = gray.copy()
            final_img[gray > final_thresh] = 255
            final_img[gray < final_thresh] = 0
            return final_img
    #show
    def otsuV(self):
        img = self.otsu(self.tmp)
        img=cv2.resize(img, (800, 800))
        cv2.imshow('image', img)
        cv2.waitKey(0)
        windows = cv2.destroyAllWindows()
    def ECC(self,I):
        etq = 1
        tableEQV = np.array(range(256))
        for i in range(len(I)):
            IMC = []
            for j in range(len(I[0])):
                if I[i][j] != 0:
                    if I[i - 1][j] == 0 and I[i][j - 1] == 0:
                        I[i][j] = etq
                        etq = etq + 1
                    if I[i - 1][j] == 0 and I[i][j - 1] != 0:
                        I[i][j] = I[i][j - 1]
                    if I[i - 1][j] != 0 and I[i][j - 1] == 0:
                        I[i][j] = I[i - 1][j]
                    if I[i - 1][j] != 0 and I[i][j - 1] != 0:
                        tmp = min(tableEQV[I[i - 1][j]], tableEQV[I[i][j - 1]])
                        I[i][j] = tmp
                        tableEQV[I[i - 1][j]] = tmp
                        tableEQV[I[i][j - 1]] = tmp
        for i in range(len(I)):
            for j in range(len(I[0])):
                if tableEQV[I[i][j]]!=I[i][j]:
                    I[i][j]=tableEQV[I[i][j]]
        l=[]
        for i in range(len(I)):
            for j in range(len(I[0])):
                if I[i][j] in l:
                    pass
                else:l.append(I[i][j])
        for k in l:
            s=0
            p=0
            for j in range(len(I[0])):
                    if k == I[0][j]:
                        s=s+1
                        p=p+1
            for i in range(1,len(I)-1):
                for j in range(len(I[0])):
                    if k == I[i][j]:
                        s=s+1
                p=p+2
            for j in range(len(I[0])):
                if k == I[len(I)-1][j]:
                    s = s + 1
                    p = p + 1

            c=(4*math.pi*s)/(p*p)
            print('Aire,Périmètre,Compacité')
            print(s,p,c)
        return I
    #show
    def segR(self):
        img=self.otsu(self.tmp)
        img=self.ECC(img)
        palette = np.random.rand(256, 1, 3) * 256
        palette = np.uint8(palette)
        imgpf = img & 0xFF
        imgpf = np.uint8(imgpf)
        dst = cv2.applyColorMap(imgpf, palette)
        # cv.imshow(fenetre,imgbin)
        dst = cv2.resize(dst, (800, 800))
        cv2.imshow("labels", dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def video(self):

        cam = cv2.VideoCapture(0)
        currentframe = 0
        while (True):
            ret, frame = cam.read()
            frame = cv2.GaussianBlur(frame, (7, 7), 1.41)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edge = cv2.Canny(frame, 15, 80)
            cv2.imshow('Canny Edge', edge)
            path='C:\\Users\\lenovo\\PycharmProjects\\TAI\\image'
            name ='image' +str(currentframe) + '.jpg'
            cv2.imwrite(os.path.join(path , name),edge)
            currentframe += 1
            if cv2.waitKey(20) == ord('q'):  # Introduce 20 milisecond delay. press q to exit.
                break
    def savePhoto(self):
        """ enregistrer l'image"""

        filename = QFileDialog.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png);;TIFF(*.tiff);;BMP(*.bmp)")[0]
        if(filename):
            cv2.imwrite(filename,self.tmp)
    def rotationImage(self):
        if self.tmp is not None:
            rotated = imutils.rotate_bound(self.tmp, 180)
            self.setPhoto(rotated)
    def flipImage(self):
        if self.tmp is not None:
            flipped = cv2.flip(self.tmp, 1)
            self.setPhoto(flipped)
    def retoure(self):
        self.tmp=self.IMG
        self.setPhoto(self.tmp)



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow=QtWidgets.QMainWindow()
    ui=Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
