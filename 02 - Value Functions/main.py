# main.py
import os
import sys
from PyQt5.QtWidgets import QApplication
from elicitation_logic import ElicitationProcess
from main_window import MainWindow


def _maybe_force_xcb_on_wayland():
    """If running under Wayland, restart the process with QT_QPA_PLATFORM=xcb
    to improve compatibility (avoids some Wayland-specific focus/activation issues).

    This is a conservative automatic fallback: it only restarts if WAYLAND display
    is present and the user hasn't explicitly set `QT_QPA_PLATFORM`.
    """
    try:
        if 'WAYLAND_DISPLAY' in os.environ and not os.environ.get('QT_QPA_PLATFORM'):
            # set to xcb so Qt uses XWayland (X11 compatibility) which avoids
            # QWindow::requestActivate() limitations on some Wayland compositors.
            os.environ['QT_QPA_PLATFORM'] = 'xcb'
            # Re-exec the current python process with same args so the platform
            # selection takes effect before Qt loads.
            python = sys.executable
            args = [python] + sys.argv
            print("Detected Wayland session — restarting with QT_QPA_PLATFORM=xcb for compatibility...")
            os.execv(python, args)
    except Exception:
        # if anything goes wrong, continue without forcing platform
        pass


if __name__ == "__main__":
    _maybe_force_xcb_on_wayland()

    app = QApplication(sys.argv)

    # Initialize the elicitation process
    process = ElicitationProcess()

    # Create and show the main window
    window = MainWindow(process)
    window.show()

    sys.exit(app.exec_())
