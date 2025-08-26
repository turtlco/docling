import threading

pypdfium2_lock = threading.Lock()
pymupdf_lock = threading.RLock()
