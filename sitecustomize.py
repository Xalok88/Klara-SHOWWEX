import sys, time, importlib

def _hook():
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:
        return
    for _ in range(200):  # ~20 сек
        try:
            app = QApplication.instance()
            if app:
                klara2 = importlib.import_module("klara2")
                klara2._start_qt_bits()
                return
        except Exception:
            pass
        time.sleep(0.1)

_hook()