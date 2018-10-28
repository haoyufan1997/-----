# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 09:31:06 2018

@author: fhy
"""


import pandas as pd
import random
from PIL import Image
import os
import shutil
from sklearn.model_selection import train_test_split
import pylab as pl
import copy
import numpy as np
from skimage.util import random_noise
class_dict = {'不导电':'defect1','擦花':'defect2','横条压凹':'defect3','桔皮':'defect4',
        '漏底':'defect5','碰伤':'defect6','起坑':'defect7','凸粉':'defect8',
        '涂层开裂':'defect9','脏点':'defect10','其他':'defect11','正常':'norm'
        }
val_ratio = 0.10
setpath = r"../data/train/瑕疵样本"
setpath1 = r"../data/train/无瑕疵样本"
path_train = r'../data/processed_train'
path_valid = r'../data/processed_valid'
path_normal = r"../data/processed_train/norm" 
index_file = []
label = []



def his(im):
    '''
    直方图均衡化处理
    '''
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]
    imhist_r,bins_r = pl.histogram(r,256,normed=True)
    imhist_g,bins_g = pl.histogram(g,256,normed=True)
    imhist_b,bins_b = pl.histogram(b,256,normed=True)
    cdf_r = imhist_r.cumsum()
    cdf_g = imhist_g.cumsum()
    cdf_b = imhist_b.cumsum()
    cdf_r = cdf_r*255/cdf_r[-1]
    cdf_g = cdf_g*255/cdf_g[-1]
    cdf_b = cdf_b*255/cdf_b[-1]
    im_r = pl.interp(r.flatten(),bins_r[:256],cdf_r)
    im_g = pl.interp(g.flatten(),bins_g[:256],cdf_g)
    im_b = pl.interp(b.flatten(),bins_b[:256],cdf_b)
    #原始通道图
    #均衡化之后的通道图
    im_r = im_r.reshape([im.shape[0],im.shape[1]])
    im_g = im_g.reshape([im.shape[0],im.shape[1]])
    im_b = im_b.reshape([im.shape[0],im.shape[1]])
    im_p = copy.deepcopy(im)
    im_p[:,:,0] = im_r
    im_p[:,:,1] = im_g
    im_p[:,:,2] = im_b
    return im_p
def read_img():
    i = 1
    for file in os.listdir(setpath):
        if file == '其他':
            path_other = r"../data/processed_train/defect11"
            os.mkdir(path_other)
            for file1 in os.listdir(os.path.join(setpath,file)):
                if file1 != '.DS_Store' and os.listdir(os.path.join(setpath,file,file1)) :
                    for img in os.listdir(os.path.join(setpath,file,file1)):
                        if img == '.DS_Store':
                            continue
                        way = os.path.join(setpath,file,file1,img)
                        img_name = file1 + str(i) + '.jpg'
                        path_img = os.path.join(path_other,img_name)#图片路径
                        index_name = img_name#图片索引
                        index_file.append(index_name)
                        label.append(file)
                        shutil.copyfile(way,path_img)
                        i += 1
        else:
            path_defect = r"../data/processed_train"
            path_defect = os.path.join(path_defect,class_dict[file])
            os.mkdir(path_defect)
            for img in os.listdir(os.path.join(setpath,file)):
                way = os.path.join(setpath,file,img)
                path_img = os.path.join(path_defect,file)
                img_name = str(i) + '.jpg'
                path_img = path_img + img_name#图片路径
                index_name = file + img_name#图片索引
                index_file.append(index_name)
                label.append(file)
                shutil.copyfile(way,path_img)
                i += 1  
    os.mkdir(path_normal)
    for img in os.listdir(setpath1):
        way = os.path.join(setpath1,img)
        img_name = 'norm' + str(i) + '.jpg'
        path_img = os.path.join(path_normal,img_name)
        index_file.append(img_name)
        label.append('norm')
        shutil.copyfile(way,path_img)
        i += 1  
#划分训练集，验证集,用sklearn分层抽样

def set_split():
    '''
    将处理好的文件 分为训练集，验证集
    '''
    random.seed(0)         
    label_file = pd.DataFrame({'index_name': index_file, 'label': label})
    all_data = label_file
    train_data_list, val_data_list = train_test_split(all_data, test_size=val_ratio, random_state=0, stratify=all_data['label'])

    index = list(val_data_list['index_name'])
    for file_set in os.listdir(path_train):
        os.mkdir(os.path.join(path_valid,file_set))
        path_valid_set = os.path.join(path_valid,file_set)#创建目录
        for img in os.listdir(os.path.join(path_train,file_set)):
            if img in index:
                old_src = os.path.join(path_train,file_set,img)
                new_src = os.path.join(path_valid_set,img) 
                os.rename(old_src,new_src)

'''
def reduce_norm():
    
    #欠采样正常类别

    random.seed(0)
    path_reduce = r'../data/norm2'
    os.mkdir(path_reduce)
    file_list = os.listdir(path_normal)
    length = len(file_list)
    index = random.sample(range(0,length),int(length*0.5))
    for i in index:
        old_src = os.path.join(path_normal,file_list[i])
        new_src = os.path.join(path_reduce,file_list[i])
        os.rename (old_src,new_src)
'''

def preprocess_train(path):
    
    i = 1
    for file in os.listdir(path):
        if file != 'defect11':
            for img in os.listdir(os.path.join(path,file)):
                path_img = os.path.join(path,file,img)
                path_file = os.path.join(path,file)
                im = Image.open(path_img)
                im = pl.array(im)
                im = his(im)
                im = Image.fromarray(im.astype('uint8')).convert('RGB')
                im_name = 'new' + str(i) + '.jpg'
                path_newim = os.path.join(path_file,im_name)
                im.save(path_newim)
                if os.path.exists(path_img):
                    os.remove(path_img)
                i += 1
def preprocess_valid(path):    
    i = 1
    for file in os.listdir(path):
        for img in os.listdir(os.path.join(path,file)):
            path_img = os.path.join(path,file,img)
            path_file = os.path.join(path,file)
            im1 = Image.open(path_img)
            image = pl.array(im1)
            newimg = his(image)
            newimg = Image.fromarray(newimg.astype('uint8')).convert('RGB')
            newimg_name = 'new' + str(i) + '.jpg'
            path_newimg = os.path.join(path_file,newimg_name)
            newimg.save(path_newimg)
            i += 1 
            if os.path.exists(path_img):
                os.remove(path_img)
def process_defect11(path):
    '''
    输入：图像路径
    输出：增强过的图像集合
    增强方法： 旋转，加噪，直方图均衡化
    扩充defect11
    '''
   
    i = 1
    path_1 = r'../data/processed_valid/defect11'
    path_2 = r'../data/processed_train/defect11'
    for img in os.listdir(path_1):
        way = os.path.join(path_1,img)
        img_name = img + 'copy'+ str(i) + '.jpg'
        path_img = os.path.join(path_2,img_name)
        index_file.append(img_name)
        shutil.copyfile(way,path_img)
        i += 1 
  
    i = 1
    #special_name = ['打白点','打磨印','返底','火山口','铝屑','喷涂划伤','气泡','托烂','纹粗','油印','油渣','杂色']
    for img in os.listdir(path):
        if '打白点' in img:
            angles = np.random.random(2)*360
        elif '打磨印' in img:
            angles = np.random.random(10)*360
        elif '返底' in img:
            angles = np.random.random(20)*360
        elif '划伤' in img:
            angles = np.random.random(2)*360
        elif '火山口' in img:
            angles = np.random.random(5)*360
        elif '铝屑' in img:
            angles = np.random.random(4)*360
        elif '喷涂碰伤' in img:
            angles = np.random.random(20)*360
        elif '气泡' in img:
            angles = np.random.random(3)*360
        elif '托烂' in img:
            angles = np.random.random(3)*360
        elif '油印' in img:
            angles = np.random.random(20)*360
        elif '油渣'in img:
            angles = np.random.random(4)*360
        elif '杂色' in img:
            angles = np.random.random(4)*360
        elif '粘接' in img:
            angles = np.random.random(5)*360
        elif '碰凹' in img:
            angles = np.random.random(2)*360
        else:
            angles = []
        path_img = os.path.join(path,img)
        path_file = path
        angles.append(0)
        angles = [int(x) for x in angles]
        im1 = Image.open(path_img)
        for angle in angles:
            image = im1.rotate(angle)
            image = pl.array(image)
            newimg = random_noise(image, mode='gaussian', clip=True)*255
            newimg = Image.fromarray(newimg.astype('uint8')).convert('RGB')
            newimg_name =  'new' + str(i) + '.jpg'
            path_newimg = os.path.join(path_file,newimg_name)
            newimg.save(path_newimg)
            i += 1
read_img()
#reduce_norm()
set_split()                       
process_defect11(r"../data/processed_train/defect11")           


'''
setpath = '../data/train/瑕疵样本/'
for file in os.listdir(setpath):
    if file == '其他':
        path_other = r"../data/processed_train/defect11"
        os.mkdir(path_other)
        for file1 in os.listdir(os.path.join(setpath,file)):
            if file1 != '.DS_Store' and os.listdir(os.path.join(setpath,file,file1)) :
                for img in os.listdir(os.path.join(setpath,file,file1)):
                    if img == '.DS_Store':
                        continue
                    way = os.path.join(setpath,file,file1,img)
                    path_img = os.path.join(path_other,file1)
                    img_name = str(i) + '.jpg'
                 
                    path_img = path_img + img_name#图片路径
                    index_name = file + img_name#图片索引
                    index_file.append(index_name)
                    label.append(file)
                    shutil.copyfile(way,path_img)
                    i += 1
'''
#process_defect11(r"../data/processed_train/defect11")
#preprocess_valid(path_valid)
'''
setpath = '../data/train/瑕疵样本/其他'
i=1
for file1 in os.listdir(setpath):
    if file1 != '.DS_Store' and os.listdir(os.path.join(setpath,file1)):
        for img in os.listdir(os.path.join(setpath,file1)):
            if img == '.DS_Store':
                continue
            way = os.path.join(setpath,file1,img)
            name = str(i) + '.jpg'
            path_new = r"../data/processed_valid/defect11"
            path_new = os.path.join(path_new,name)
            shutil.copyfile(way,path_new)
            i += 1
                           
    
                        
                

            
lis = os.listdir(r"../data/processed_valid/defect11")
index = random.sample(range(0,len(lis)),int(len(lis)*0.8))
for i in index:
    os.remove(os.path.join(r"../data/processed_valid/defect11",lis[i]))
''' 
def preprocess_train(path):
    
    i = 1
    for file in os.listdir(path):
         for img in os.listdir(os.path.join(path,file)):
            if '!' in img:
                angles = np.random.random(10)*360
            elif '碰凹' in img:
                angles = np.random.random(1)*360
            angles = [int(x) for x in angles]
            im = Image.open(path_img)
            for angle in angles:    
                image = im1.rotate(angle)
                newimg = pl.array(image)
                newimg = random_noise(newimg, mode='gaussian', clip=True)*255
                newimg = Image.fromarray(newimg.astype('uint8')).convert('RGB')
                newimg_name =  'new' + str(i) + '.jpg'
                path_newimg = os.path.join(path_file,newimg_name)
                newimg.save(path_newimg)
                i += 1
                
preprocess_train(path_train)                        


           
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
