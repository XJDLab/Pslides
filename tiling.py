import openslide
import os
import numpy as np
import cv2
from skimage import io

## code to generate IF tiles (the size and downsample level of the tiles can be adjusted)
scr= #directory saving IF WSI files
dst= #directory to save generated tiles
with open (r"path to the txt file saving the annotation information ", mode='r', encoding='utf-8') as f:
    for line in f.readlines():
        filename=line.split('\t')[0].strip('\n')
        print('processing{}'.format(filename))
        #print(filename)
        start_x=int(line.split('\t')[1].strip('\n'))
        #print(start_x)
        start_y=int(line.split('\t')[2].strip('\n'))
        #print(start_y)
        w=int(int(line.split('\t')[3].strip('\n')))
        #print(w)
        h=int(int(line.split('\t')[4].strip('\n')))
        #print(h)
        scr_path=os.path.join(scr,filename+'.mrxs')
        dst_dir=os.path.join(dst,filename,'TILES_10X_ds1') #save tile at downsample level1; the level is adjustable according to the difficulty level of alignment
        slide=openslide.OpenSlide(scr_path)
        for y in range(start_y,start_y+h,10240): #10240 represents the size of 10x tiles
            for x in range(start_x,start_x+w,10240):
                patch_PIL = np.array(slide.read_region((x,y), 1, (5120, 5120)).convert('RGB')) #1 represents downsample level1, and is calculated by the size of tiles at level 0 and the downsample level
                dst_path=os.path.join(dst_dir,"{}_{}_{}.jpg".format(filename,int((y-start_y)/10240),int((x-start_x)/10240)))
                cv2.imwrite(dst_path,patch_PIL)


## code to generate IHC or HE tiles using the information from the generated IF tiles

scr= #directory saving IHC or HE WSI files
dst=  #directory to save generated IHC or HE tiles
IF_tiles= #directory  saves generated tiles
with open (r"path to the txt file saving the annotation information ", mode='r', encoding='utf-8') as f:
    for line in f.readlines():
        filename=line.split('\t')[0].strip('\n')
        #print('processing{}'.format(filename))
        #print(filename)
        start_x=int(line.split('\t')[5].strip('\n'))
        #print(start_x)
        start_y=int(line.split('\t')[6].strip('\n'))
        #print(start_y)
        scr_path=os.path.join(scr,filename+'.mrxs')
        #dst_dir=os.path.join(dst,filename,'tiles')
        slide=openslide.OpenSlide(scr_path)
        for file in os.listdir(os.path.join(IF_tiles,filename,'TILES_10X_d1')):#save tile at downsample level1; the level is adjustable according to the difficulty level of alignment
            ind_x=int(os.path.splitext(file)[0].split('_')[2])
            ind_y=int(os.path.splitext(file)[0].split('_')[1])
            x=start_x+ind_x*11560-7630 # to generate approximately double size tiles to the IF tiles
            y=start_y+ind_y*11560-7630 # to generate approximately double size tiles to the IF tiles
            patch_PIL = np.array(slide.read_region((x,y), 1, (11750, 11750)).convert('RGB'))
            dst_path=os.path.join(dst,filename,'TILES_10X_d2',"{}_{}_{}.jpg".format(filename,ind_y,ind_x))
            io.imsave(dst_path,patch_PIL)