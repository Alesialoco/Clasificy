from PyQt5 import QtCore, QtGui, QtWidgets


class UiDialog(object):
    def setup_ui(self, dialog):
        dialog.setObjectName("Dialog")
        dialog.resize(1000, 800)
        self.pb = QtWidgets.QPushButton(dialog)
        self.pb.setGeometry(QtCore.QRect(840, 30, 141, 51))
        self.pb.setObjectName("pushButton")
        self.text = QtWidgets.QPlainTextEdit(dialog)
        self.text.setGeometry(QtCore.QRect(30, 170, 931, 611))
        self.text.setObjectName("plainTextEdit")
        self.line = QtWidgets.QLineEdit(dialog)
        self.line.setGeometry(QtCore.QRect(20, 30, 791, 51))
        self.line.setObjectName("lineEdit")
        self.label = QtWidgets.QLabel(dialog)
        self.label.setGeometry(QtCore.QRect(430, 110, 291, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.retranslate_ui(dialog)
        QtCore.QMetaObject.connectSlotsByName(dialog)

    def retranslate_ui(self, dialog):
        _translate = QtCore.QCoreApplication.translate
        dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pb.setText(_translate("Dialog", "Загрузить"))
        self.label.setText(_translate("Dialog", "Классификация"))
