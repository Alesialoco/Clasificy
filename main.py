import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtCore, QtGui, QtWidgets


class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(1000, 800)
        self.setWindowTitle("Классификатор")
        self.pb = QtWidgets.QPushButton("Загрузить", self)
        self.pb.setGeometry(QtCore.QRect(840, 30, 141, 51))
        self.text = QtWidgets.QPlainTextEdit(self)
        self.text.setGeometry(QtCore.QRect(30, 170, 931, 611))
        self.line = QtWidgets.QLineEdit(self)
        self.line.setGeometry(QtCore.QRect(20, 30, 791, 51))
        self.label = QtWidgets.QLabel("Классификация", self)
        self.label.setGeometry(QtCore.QRect(430, 110, 291, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        QtCore.QMetaObject.connectSlotsByName(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec_())
