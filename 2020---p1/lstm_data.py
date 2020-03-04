# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:00:06 2019

@author: jason
"""

import os
import re
import io
import  os
import time
import datetime
#import codecs

top_dir = "E:\\Code\\AVISO_dat2\\"
dst_dir =  "E:\\Code\\"


def TimeStampToTime(timestamp):
     timeStruct = time.localtime(timestamp)
     return time.strftime('%Y-%m-%d %H:%M:%S',timeStruct)

def trim_dir(path):
    print("目录：" + path)
    for root, dirs, files in os.walk(path):
        fp_dst = open(dst_dir+datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')+'.csv', 'a+')
        for name in files:
            try:
                #print('ProcessFile==='+os.path.join(root, name))
                date_id = name[7:15]
                #print(date_id)
                if date_id[4:4]==9:
                    print('ProcessFile==='+date_id)
               
               #with  open(dst_dir, 'a+') as  fp_dst:
                    #fp_dst.write(date_id+","+line)
                    
                line = trim_file(os.path.join(root, name))

                
                fp_dst.write(date_id+","+line)
                print(date_id + "," + line)
            except:
                print('WrongFile==='+os.path.join(root, name))
        # for name in dirs:
        #     trim_dir(os.path.join(root, name))
          #finally:
        fp_dst.close()

def trim_file(path):
     line = ''
     with  io.open(path, 'r') as  fp_src:
        line = fp_src.readline() 
        
     return line
    
    
trim_dir(top_dir)