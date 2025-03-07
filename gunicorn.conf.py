import multiprocessing

bind = "0.0.0.0:8888"
workers = 2
worker_class = "gthread"
threads = 4
timeout = 300
preload_app = False  # Fundamental para CUDA