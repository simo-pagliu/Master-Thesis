# main.py
import sys
from PyQt5.QtWidgets import QApplication
from elicitation_logic import ElicitationProcess
from main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Initialize the elicitation process
    process = ElicitationProcess()

    # Create and show the main window
    window = MainWindow(process)
    window.show()

    sys.exit(app.exec_())
