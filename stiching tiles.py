import os
import cv2
import numpy as np

## using the coordinate in the file name to get stiched tiles (a 10x example)

scr= # path saving tiles to be stiched
dst= #path for the stiched tiles to be saved
splicing_pic = np.zeros((10240, 10240, 3)) #10x tiles
for y in range(10):
    for x in range(10):
        indx=str(y*10+x)
        img_path=os.path.join(scr,"IF1_"+indx+".png") #IF1_ can be changed according to the file name
        img=cv2.imread(img_path)
        splicing_pic[y*1024:y*1024+1024,x*1024:x*1024+1024]=img
    dst_path=os.path.join(dst,"he.png")
    cv2.imwrite(dst_path,splicing_pic)