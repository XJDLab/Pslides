import numpy as np
import cv2
import os
from skimage import data, exposure, img_as_float

def matchKeypoints( kpsA, kpsB, featuresA, featuresB,ratio, reprojThresh):

    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []

    for m in rawMatches:

        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    if len(matches) > 4:

        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)

        return (matches, H, status)

    return None 



def detectAndDescribe(image):

    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    descriptor = cv2.SIFT_create()

    (kps, features) = descriptor.detectAndCompute(image, None)

    kps = np.float32([kp.pt for kp in kps])

    return kps, features



scr_gray_dir=  #directory saving grayscale IF tiles after autocontrast and autolevel correction
scr_he_dir= #directory saving HE or IHC tiles
with open (r"path to the txt file saving the annotation information ", mode='r', encoding='utf-8') as f:
    for line in f.readlines():
        filename=line.split('\t')[0].strip('\n')
        print('processing{}'.format(filename))
        scr_gray=os.path.join(scr_gray_dir,filename,'TILES_AC')
        #scr_he_hed=os.path.join(scr_he_dir,filename,'TILES_HED')
        scr_he=os.path.join(scr_he_dir,filename,'TILES_10X_d1')
        dst=os.path.join(scr_he_dir,filename,'TILES_ALIGNED_ds1')
        for file in os.listdir(scr_gray):
            img_align=cv2.imread(os.path.join(scr_he,file))
            #img_align_hed=cv2.imread(os.path.join(scr_he_hed,file))
            
            img_temp=cv2.imread(os.path.join(scr_gray,file))
            #img1 = cv2.cvtColor(img_align_hed, cv2.COLOR_BGR2GRAY)
            img1 = cv2.cvtColor(img_align, cv2.COLOR_BGR2GRAY)
            img1=exposure.adjust_gamma(img1, 3.0) 
            #img2=img_temp
            img2 = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
            img2=exposure.adjust_gamma(img2, 3.0) 
            img1s=cv2.resize(img1,(2350,2350)) # to save memory
            img2s=cv2.resize(img2,(1280,1280)) # to save memory
            height, width = img2s.shape
            #height, width = img2.shape
            #kpsA, featuresA=detectAndDescribe(img1s)
            #kpsB, featuresB=detectAndDescribe(img2s)
            kpsA, featuresA=detectAndDescribe(img1s)
            kpsB, featuresB=detectAndDescribe(img2s)
            try:
                matches, H, status=matchKeypoints(kpsA, kpsB, featuresA, featuresB,0.75, 1.0)
                transformed_img = cv2.warpPerspective(img1s, H, (width, height))
                dst_path=os.path.join(dst,file)
                cv2.imwrite(dst_path, transformed_img)
            except:
                pass
            continue 