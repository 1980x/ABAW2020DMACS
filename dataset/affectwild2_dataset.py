'''
Aum Sri Sai Ram

Implementation of Aff-Wild2 dataset class

Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 28-09-2020
Email: darshangera@sssihl.edu.in
'''


import torch.utils.data as data
from PIL import Image, ImageFile
import os
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage import io
from torchvision import transforms
import random
ImageFile.LOAD_TRUNCATED_IAMGES = True
import pickle
import random

# Labels ['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise']:0,1,2,3,4,5,6

def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)

def switch_expression(expression_argument):
    switcher = {
         0:'Neutral',
         1:'Happiness',
          2: 'Sadness',
        3: 'Surprise',
4: 'Fear', 5: 'Disgust', 6: 'Anger',
    }
    return switcher.get(expression_argument, 0) #default neutral expression

def change_emotion_label_same_as_affectnet(emo_to_return):
        """
        Parse labels to make them compatible with AffectNet.  
        #https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/model/utils/udata.py
        """

        if emo_to_return == 0:
            emo_to_return = 0
        elif emo_to_return == 1:
            emo_to_return = 6
        elif emo_to_return == 2:
            emo_to_return = 5
        elif emo_to_return == 3:
            emo_to_return = 4
        elif emo_to_return == 4:
            emo_to_return = 1
        elif emo_to_return == 5:
            emo_to_return = 2
        elif emo_to_return == 6:
            emo_to_return = 3

        return emo_to_return



def _read_path_label(_train_mode, file_path):
        
        data = pickle.load(open(file_path, 'rb'))
        # read frames ids
        if _train_mode == 'Train':
            data = data['EXPR_Set']['Training_Set']
        elif _train_mode == 'Validation':
            data = data['EXPR_Set']['Validation_Set']
        elif _train_mode == 'Test':
            data = data['EXPR_Set']['Test_Set']
        else:
            raise ValueError("train mode must be in : Train, Validation, Test")
        return data

def default_reader(fileList,  num_classes=7, train_mode ='Train'):
    imgList = []
   
    start_index = 1
    if train_mode =='Train':
       max_samples = 9000000            #Change it to reduce number of samples per class for Affwild2 training set
    else:
       max_samples = 9000000
    num_per_cls_dict = dict()
    for i in range(0, num_classes):
        num_per_cls_dict[i] = 0
    
    expression_0, expression_1,expression_2, expression_3, expression_4,expression_5,expression_6 = 0,0,0,0,0,0,0
    data_dict = _read_path_label(train_mode, fileList)
    
    all_list = []         
    if  train_mode in ['Train','Validation']: #training or validation
       if train_mode =='Validation':
            for video_name in data_dict.keys(): #Each video is a key            
                frame_dict  = data_dict[video_name]
                labels, imagepaths = frame_dict['label'],frame_dict['path']
                for i in range(len(labels)):
                    imagename,label = imagepaths[i][2:],labels[i] #2: is to remove ./ in ./cropped_aligned/48-30-720x1280/01589.jpg 0
                    all_list.append([imagename,label])
       elif train_mode =='Train':
            for video_name in data_dict.keys(): #Each video is a key            
                frame_dict  = data_dict[video_name]
                labels, imagepaths = frame_dict['label'],frame_dict['path']
                for i in range(len(labels)):
                    imagename,label = imagepaths[i][2:],labels[i] #2: is to remove ./ in ./cropped_aligned/48-30-720x1280/01589.jpg 0
                    all_list.append([imagename,label])
                    #print(len(all_list), all_list[-3:]) 

            #Adding below valid to train
            """
            data_dict = _read_path_label('Validation', fileList)
            for video_name in data_dict.keys(): #Each video is a key            
                frame_dict  = data_dict[video_name]
                #print(frame_dict.keys())#keys: ['label', 'path', 'frames_ids']
                labels, imagepaths = frame_dict['label'],frame_dict['path']
                for i in range(len(labels)):
                    imagename,label = imagepaths[i][2:],labels[i] #2: is to remove ./ in ./cropped_aligned/48-30-720x1280/01589.jpg 0
                    all_list.append([imagename,label])
                    #print(len(all_list), all_list[-3:])
		    """
           
    elif train_mode =='Test': #test
         for video_name in data_dict.keys(): #Each video is a key                            
                frame_dict  = data_dict[video_name]                
                labels, imagepaths = frame_dict['label'], frame_dict['path']
                for i in range(len(labels)):
                    imagename,label = imagepaths[i][2:],labels[i] #2: is to remove ./ in ./cropped_aligned/48-30-720x1280/01589.jpg 0
                    all_list.append([imagename,label])
    else:
           print('Not implemented yet.\n')

    random.shuffle(all_list)


    for i in range(len(all_list)):
            imgPath, expression =all_list[i]

            expression = change_emotion_label_same_as_affectnet(expression) 

            if expression == 0:
               expression_0 = expression_0 + 1            
               if expression_0 > max_samples:
                  continue
  
            if expression == 1:
               expression_1 = expression_1 + 1
               if expression_1 > max_samples:
                  continue  

            if expression == 2:
               expression_2 = expression_2 + 1
               if expression_2 > max_samples:
                  continue  

            if expression == 3:
               expression_3 = expression_3 + 1
               if expression_3 > max_samples:
                  continue  

            if expression == 4:
               expression_4 = expression_4 + 1
               if expression_4 > max_samples:
                  continue  

            if expression == 5:
               expression_5 = expression_5 + 1
               if expression_5 > max_samples:
                  continue  

            if expression == 6:
               expression_6 = expression_6 + 1
               if expression_6 > max_samples:
                  continue  

            imgList.append([imgPath, expression])
            num_per_cls_dict[expression] = num_per_cls_dict[expression] + 1 
        
    print(train_mode, ' has total included: ', len(imgList), ' with split \t', num_per_cls_dict)
    return imgList,num_per_cls_dict

   
class ImageList(data.Dataset):
    def __init__(self, root, fileList, train_mode = 'Validation', transform=None,  list_reader=default_reader, loader=PIL_loader):
        self.root = root
        self.cls_num = 7
        self.imgList, self.num_per_cls_dict =  list_reader(fileList, self.cls_num, train_mode)
        self.transform = transform
        self.loader = loader
        self.fileList  = fileList
        self.train_mode = train_mode

    def __getitem__(self, index):
        imgPath,  target_expression = self.imgList[index]
        
        imagefullpath = os.path.join(self.root, imgPath)

        if not os.path.exists(imagefullpath) and self.train_mode == 'Test':
           return None, imgPath         

        face = self.loader(os.path.join(self.root, imgPath))       
        
        if self.transform is not None:
            face = self.transform(face)
       
        if self.train_mode == 'Test':
           return  face, imgPath 
        else:
           return  face, target_expression 
    def __len__(self):
        return len(self.imgList)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

if __name__=='__main__':
   
   #filelist, num_per_cls_dict = default_reader('../data/Affwild2/Annotations/annotations.pkl', 7, train_mode='Validation')
   filelist, num_per_cls_dict = default_reader('../data/Affwild2/Annotations/test_set.pkl', 7, train_mode='Test')
   print(len(filelist))

   rootfolder= '../data/Affwild2/'
   
   
   transform = transforms.Compose([

                              transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]),
                            ])
  
   #dataset = ImageList(rootfolder, '../data/Affwild2/Annotations/annotations.pkl', 'Validation',transform)
   dataset = ImageList(rootfolder, '../data/Affwild2/Annotations/test_set.pkl', 'Test',transform)
   
   fdi = iter(dataset)
   img_list = []
   target_list = []
   for i, data in enumerate(fdi):
       if i < 200000:
          #print(i, len(data), data[0].size(), data[1])
          img_list.append(data[1].split('/')[-1])
          continue
       else:
          break
   img_list.sort()
   print(len(img_list), img_list)
  
   

