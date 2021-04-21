from decord import VideoReader
import numpy as np
from decord import VideoReader
from decord import cpu, gpu
from utils.datasets import  letterbox
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size
import multiprocessing
import cv2
# from yolov5_original.detect import YOLOV5
from multiprocessing import Pool
import time

class Preprocess_Video(object):
    def __init__(self,url,num_threads=1,batch=64):
        self.num_threads= multiprocessing.cpu_count()
        print("cpu count ",self.num_threads)
        self.vr= VideoReader(url, ctx=cpu(0))
        self.img_size=640
        # self.detection = YOLOV5()
        self.n=len(self.vr)
        self.batch = batch

    def letterbox(self,begin, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        end=min(begin + self.batch,self.n)
        img=self.vr[begin].asnumpy()
        print("A")
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup: 
            r = min(r, 1.0)
        ratio = r, r 
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1] 
        if auto:  
            dw, dh = np.mod(dw, 32), np.mod(dh, 32) 
        elif scaleFill: 
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0] 
        dw /= 2  
        dh /= 2
        res_imgs=[]
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        for i in range(begin,end):
            print(i)
            if shape[::-1] != new_unpad:  # resize
                img=self.vr[i].asnumpy()
                img=cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
                res_imgs.append(cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)[:, :, ::-1].transpose(2, 0, 1))  # add border
        return res_imgs, ratio, (dw, dh)

    def process_frame(self):
        arg_lists = list(range(0,self.n,self.batch))  
        print(arg_lists)
        pool = Pool(processes=self.num_threads)
        result = pool.map(call_method, arg_lists)
        print(result)
        #res=detection.detect(imgs)


def call_method(i):
    # Implicitly use global copy of my_instance, not one passed as an argument
    return my_instance.letterbox(i)    
    
my_instance = Preprocess_Video("town.avi")


if __name__ == '__main__':
    t1=time.time()
    my_instance.process_frame()
    print("time : ",time.time()-t1)