import multiprocessing
import os

bind = os.environ.get('SERVER')

workers = multiprocessing.cpu_count()
          # * 2 + 1
# workers=1
max_requests = 1000