import multiprocessing
import os

bind = '0.0.0.0:5000'

workers = multiprocessing.cpu_count()
# workers = 1

          # * 2 + 1
# workers=1
# max_requests = 1000