from decord import VideoReader
import numpy as np
from decord import VideoReader
from decord import cpu, gpu
from utils.datasets import  letterbox
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size
import cv2
from yolov5_original.detect import YOLOV5
img_size=640
detection = YOLOV5()
vr = VideoReader('town.avi', ctx=cpu(0))
n=len(vr)
batch = 64
import time


def letterbox(begin,end, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    global vr
    img=vr[begin].asnumpy()
    
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
        if shape[::-1] != new_unpad:  # resize
            img=vr[i].asnumpy()
            img=cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            res_imgs.append(cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)[:, :, ::-1].transpose(2, 0, 1))  # add border
    return res_imgs, ratio, (dw, dh)

def process_frame(begin,end):
    t1=time.time()
    imgs = letterbox(begin,end, new_shape=img_size)[0]
    print("time prepare process ",time.time()-t1)
    t1=time.time()
    res=detection.detect(imgs)
    print("time detection ",time.time()-t1)
    # print(res)

    
print(n)
start=time.time()
for i in range(0,n,batch):
    t1=time.time()
    process_frame(i,min(i+batch,n))
    print(time.time()-t1)

print("time all ",time.time()-start)
# img = process_frame(frame)

# img = torch.from_numpy(img).to(self.device).float()


