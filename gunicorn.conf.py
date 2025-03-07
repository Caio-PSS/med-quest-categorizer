import multiprocessing

workers = 2                  # Igual ao número de GPUs
threads = 8                  # 8 threads por worker
timeout = 300
preload_app = True           # Pré-carregar modelo antes de forking
max_requests = 1000          # Reiniciar workers periodicamente
max_requests_jitter = 50