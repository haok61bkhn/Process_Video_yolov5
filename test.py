import os
import time
from multiprocessing import Pool
import numpy as np

class MyClass(object):
    def __init__(self):
        # self.myAttribute = os.urandom(1024*1024*1024) # basically a big memory struct(~1GB size)
        pass

    def my_multithreaded_analysis(self):
        arg_lists = list(range(4))  # Don't pass self
        pool = Pool(processes=4)
        result = pool.map(call_method, arg_lists)
        print(result)

    def analyze(self, i):
        time.sleep(10)
        return i ** 2

def call_method(i):
    # Implicitly use global copy of my_instance, not one passed as an argument
    return my_instance.analyze(i)

# Constructed globally and unconditionally, so the instance exists
# prior to forking in commonly accessible location
my_instance = MyClass()


if __name__ == '__main__':
    my_instance.my_multithreaded_analysis()