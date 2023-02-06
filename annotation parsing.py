import os
from xml.etree import ElementTree as ET


def four_points(HE_dir,IF_dir,filename):  # HE_dir and IF_dir: dirs for saving labelled xml files
    HE_path=os.path.join(HE_dir,filename+'.xml')
    IF_path=os.path.join(IF_dir,filename+'.xml')
    HE_tree=ET.parse(HE_path)
    HE_root = HE_tree.getroot()
    IF_tree=ET.parse(IF_path)
    IF_root = IF_tree.getroot()
    HE_start_x=int(float(HE_root[0][0][0][0].attrib['X']))
    HE_start_y=int(float(HE_root[0][1][0][0].attrib['Y']))
    IF_start_x=int(float(IF_root[0][0][0][0].attrib['X']))
    IF_start_y=int(float(IF_root[0][1][0][0].attrib['Y']))
    IF_end_x=int(float(IF_root[0][2][0][0].attrib['X']))
    IF_end_y=int(float(IF_root[0][3][0][0].attrib['Y']))
    w=IF_end_x-IF_start_x
    h=IF_end_y-IF_start_y
    return filename,IF_start_x,IF_start_y,w,h,HE_start_x,HE_start_y


HE_dir=  #dirs for saving labelled HE or IHCxml files
IF_dir=  #dirs for saving labelled IF xml files
path_txt= #path for txt file to save the parsed annotation information
for file in os.listdir(IF_dir):
    if os.path.splitext(file)[1] == '.xml':
        try:
            filename=file.split('.')[0]
            #filename=four_points(HE_dir,IF_dir,filename)[0]
            IF_start_x=str(four_points(HE_dir,IF_dir,filename)[1])
            IF_start_y=str(four_points(HE_dir,IF_dir,filename)[2])
            w=str(four_points(HE_dir,IF_dir,filename)[3])
            h=str(four_points(HE_dir,IF_dir,filename)[4])
            HE_start_x=str(four_points(HE_dir,IF_dir,filename)[5])
            HE_start_y=str(four_points(HE_dir,IF_dir,filename)[6])
            with open('path for txt file' ,mode='a') as out_file:   #创建文件
                out_file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(filename,IF_start_x,IF_start_y,w,h,HE_start_x,HE_start_y))
        except:
            pass
        continue