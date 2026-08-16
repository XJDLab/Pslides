import cv2
import numpy as np
import os

def autoContrast(img, cutoff):  
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    Sin = np.percentile(img, q=cutoff)  
    Hin = np.percentile(img, q=100 - cutoff) 
    
    difIn = Hin - Sin
    V1 = np.array([(min(max(255 * (i-Sin)/difIn,0), 255)) for i in range(256)])
    
    gradMed = np.median(img) 
    Mt = V1[int(gradMed)] / 160. 
    V2 = 255 * np.power(V1/255, 1/Mt)  
     
    Sout, Hout = 5, 250  
    difOut = Hout - Sout
    table = np.array([(min(max(Sout + difOut*V2[i]/255, 0), 255)) for i in range(256)]).astype("uint8")
    imgTone = cv2.LUT(img, table)
    return imgTone


def levelAdjust(img, Sin=0, Hin=255, Mt=1.0, Sout=0, Hout=255):
    Sin = min(max(Sin, 0), Hin-2)  
    Hin = min(Hin, 255)  
    Mt  = min(max(Mt, 0.01), 9.99)  
    Sout = min(max(Sout, 0), Hout-2)  
    Hout = min(Hout, 255)  

    difIn = Hin - Sin
    difOut = Hout - Sout
    table = np.zeros(256, np.uint16)
    for i in range(256):
        V1 = min(max(255 * (i-Sin)/difIn,0), 255)  
        V2 = 255 * np.power(V1/255, 1/Mt)  
        table[i] = min(max(Sout+difOut*V2/255, 0), 255)  

    imgTone = cv2.LUT(img, table)
    return imgTone