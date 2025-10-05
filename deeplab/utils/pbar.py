import os
import json
import multiprocessing as mp
from threading import Thread, Lock
from tqdm import tqdm
from time import sleep
from deeplab.utils.fileio import save_json


class ProgressBar():
    
    def __init__(self, total):

        self.cur_steps = 0
        self.all_steps = total
        self.lock = Lock()
        self.stop = False
        self.still_listen = True
        self.still_update = True

        # queues that can be shared between async processes
        m = mp.Manager()
        self.count_queue = m.Queue()
        self.error_queue = m.Queue()
        # 开始计数
        self.begin()
        return


    def begin(self):
        t1 = Thread(target=self.listen, daemon=True)
        t1.start()
        t2 = Thread(target=self.update, daemon=True)
        t2.start()
        return 
        

    def close(self):
        #with self.lock:
        self.stop = True
        while self.still_listen or self.still_update:
            sleep(1)
        return
        

    def listen(self):

        def listen_once():
            sleep(1)
            with self.lock:
                while not self.count_queue.empty():
                    data = self.count_queue.get()
                    self.cur_steps += int(data)

        while True:
            listen_once()
            if self.stop: 
                listen_once()
                break

        #with self.lock:
        self.still_listen = False

        return 

    def update(self):

        def update_once():
            sleep(1)
            # 更新进度条并休眠等待
            with self.lock:
                pbar.update(self.cur_steps - self.show_steps)
                self.show_steps = self.cur_steps
                
        self.show_steps = 0
        with tqdm(total=self.all_steps, ncols=70) as pbar:
            while True:  
                update_once()
                if self.stop and self.still_listen==False:
                    # 最后一次更新进度条后退出
                    update_once()         
                    break 
                           
        #with self.lock:
        self.still_update = False
        
        return
    
    def save_error_to_json(self, path):
        results = []
        for _ in range(self.error_queue.qsize()):
            data = self.error_queue.get()
            results.append(data)

        if len(results) > 0:
            save_json(path, results)
            print('warnings: errors ({}) happened in multiprocessing tasks.'.format(len(results)))
            return 1
            
        return 0
