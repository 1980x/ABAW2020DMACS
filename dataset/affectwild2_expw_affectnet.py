'''
Aum Sri Sai Ram


Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 28-09-2020
Email: darshangera@sssihl.edu.in
Implementation of dataset class for AffectWild2 + EXPW + AffectNet datasets

Note: Oversampling is not used in this case.

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
 
def change_emotion_label_of_expw_same_as_affectnet(emo_to_return):
    """ #EXPW emotion labels are : 	"0" "angry" 	"1" "disgust" 	"2" "fear" 	"3" "happy" 	"4" "sad"	"5" "surprise" 	"6" "neutral"   	"""
    if emo_to_return == 0:
            emo_to_return = 6
    elif emo_to_return == 1:
            emo_to_return = 5
    elif emo_to_return == 2:
            emo_to_return = 4
    elif emo_to_return == 3:
            emo_to_return = 1
    elif emo_to_return == 4:
            emo_to_return = 2
    elif emo_to_return == 5:
            emo_to_return = 3
    elif emo_to_return == 6:
            emo_to_return = 0

    return emo_to_return 
            
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
        
def default_reader(fileList,  num_classes=7, train_mode ='Train'):
    imgList = []
   
    start_index = 1
    if train_mode =='Train':
       max_samples = 20000
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
                    all_list.append([imagename,'affwild2',label])

       elif train_mode =='Train':
            for video_name in data_dict.keys(): #Each video is a key            
                frame_dict  = data_dict[video_name]

                labels, imagepaths = frame_dict['label'],frame_dict['path']
                for i in range(len(labels)):
                    imagename,label = imagepaths[i][2:],labels[i] #2: is to remove ./ in ./cropped_aligned/48-30-720x1280/01589.jpg 0
                    all_list.append([imagename,'affwild2',label])
                    

            #Adding below valid to train
            """
            data_dict = _read_path_label('Validation', fileList)
            for video_name in data_dict.keys(): #Each video is a key            
                frame_dict  = data_dict[video_name]
                #print(frame_dict.keys())#keys: ['label', 'path', 'frames_ids']
                labels, imagepaths = frame_dict['label'],frame_dict['path']
                for i in range(len(labels)):
                    imagename, label = imagepaths[i][2:],labels[i] #2: is to remove ./ in ./cropped_aligned/48-30-720x1280/01589.jpg 0
                    all_list.append([imagename,'affwild2',label])
                    #print(len(all_list), all_list[-3:])
            """
    elif train_mode =='Test': #test
         for video_name in data_dict.keys(): #Each video is a key                            
                frame_dict  = data_dict[video_name]                
                labels, imagepaths = frame_dict['label'], frame_dict['path']
                for i in range(len(labels)):
                    imagename,label = imagepaths[i][2:],labels[i] #2: is to remove ./ in ./cropped_aligned/48-30-720x1280/01589.jpg 0
                    all_list.append([imagename,'affwild2', label])
    else:
           print('Not implemented yet.\n')

    random.shuffle(all_list)


    for i in range(len(all_list)):
            imgPath,name,  expression =all_list[i]

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

            imgList.append([imgPath, 'affwild2',expression])
            num_per_cls_dict[expression] = num_per_cls_dict[expression] + 1 
        
    print(train_mode, ' has total included: ', len(imgList), ' with split \t', num_per_cls_dict)
    return imgList,num_per_cls_dict        


def default_reader_expw(fileList):
    
    counter_loaded_images_per_label = [0 for _ in range(7)]

    imgList = []
    if fileList.find('Expw_metafile_aligned.txt') > -1:
 
       fp = open(fileList,'r')

       for names in fp.readlines():
           image_path, target  = names.split(' ')  #Eg. for each entry before underscore lable and afterwards name in 1_fer0034656.png 8 0, 2_fer0033878.png 8 0

           target = change_emotion_label_of_expw_same_as_affectnet(int(target))
           
           
           if counter_loaded_images_per_label[target] >= 3000:
              continue
           else:
              counter_loaded_images_per_label[target] += 1 
              imgList.append((image_path,'expw', int(target),)) #'0000-bbox' is added to make same number of elements in each tuple from both dataset

       fp.close()
       
       print(fileList, ' has total: ',sum(counter_loaded_images_per_label), counter_loaded_images_per_label)
       
       return imgList 
       

def default_reader_affectnet(fileList):
    imgList = []
    if fileList.find('validation.csv')>-1: #hardcoded for Affectnet dataset
       start_index = 0
       max_samples = 100000
    else:
       start_index = 1
       max_samples = 10000

    num_per_cls_dict = dict()
    for i in range(0, 7):
        num_per_cls_dict[i] = 0
    counter_loaded_images_per_label = [0 for _ in range(7)]

    expression_0 = 0
    expression_1 = 0
    expression_2 = 0
    expression_3 = 0
    expression_4 = 0
    expression_5 = 0
    expression_6 = 0
    expression_7 = 0
    
 
    if fileList.find('training') > -1 or fileList.find('validation') > -1:     

        fp = open(fileList, 'r')
        for line in fp.readlines()[start_index:]:  #Ist line is header for automated labeled images
            
            imgPath  = line.strip().split(',')[0] #folder/imagename
            (x,y,w,h)  = line.strip().split(',')[1:5]#bounding box coordinates
            
            expression = int(line.strip().split(',')[6])#emotion label

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

            if expression == 7:
               expression_7 = expression_7 + 1
               if expression_7 > max_samples:
                  continue  

            if fileList.find('training') > -1 and expression not in [ 0,1,2,3, 6, 7, 8,9,10]: #Adding only list of fear and disgust expressions
               imgList.append([imgPath,'affectnet', expression])
               num_per_cls_dict[expression] = num_per_cls_dict[expression] + 1
            elif fileList.find('validation') > -1 and expression not in [ 7, 8,9,10]:  
               imgList.append([imgPath,'affectnet', expression])
               num_per_cls_dict[expression] = num_per_cls_dict[expression] + 1
        fp.close()
        print('Affectnet total included: ', len(imgList), ' class wise: ', num_per_cls_dict)
        return imgList,num_per_cls_dict
       
class ImageList(data.Dataset):
    def __init__(self, root, fileList, train_mode = 'Validation', transform=None,  list_reader=default_reader, loader=PIL_loader):
        self.root = root
        self.cls_num = 7
        
        if train_mode is 'Train':
           self.imgList_affwild2, self.num_per_cls_dict =  list_reader(fileList, self.cls_num, train_mode)
           self.imgList_affectnet1,  self.num_per_cls_dict = default_reader_affectnet('../data/Affectnetmetadata/training.csv')
           self.imgList_affectnet2,  self.num_per_cls_dict = default_reader_affectnet('../data/Affectnetmetadata/validation.csv')
           self.imgList_expw = default_reader_expw('../data/ExpW/data/label/Expw_metafile_aligned.txt')             
           self.imgList =  self.imgList_affwild2 + self.imgList_affectnet1 + self.imgList_affectnet2+ self.imgList_expw
           print(len(self.imgList_affwild2), len(self.imgList_affectnet1),len(self.imgList_affectnet2), len(self.imgList_expw), len(self.imgList))

        else:
           self.imgList, self.num_per_cls_dict =  list_reader(fileList, self.cls_num, train_mode)
            
           
        self.transform = transform
        self.loader = loader
        self.fileList  = fileList
        self.train_mode = train_mode

    def __getitem__(self, index):
    
        imgPath, dataset,  target_expression = self.imgList[index]
 
        
        if dataset == 'affectnet':                   
           img = self.loader(os.path.join('../data/AffectNetdataset/Manually_Annotated_Images_aligned/', imgPath))               
        elif dataset == 'expw':        
           img = self.loader(os.path.join('../data/ExpW/data/image_aligned/', imgPath))
        else:
           imagefullpath = os.path.join(self.root, imgPath)
           if not os.path.exists(imagefullpath) and self.train_mode == 'Test':
              return None, imgPath         
           img = self.loader(os.path.join(self.root, imgPath))
                                   
        if self.transform is not None:
           img = self.transform(img)
        
        if self.train_mode == 'Test':
           return  img, imgPath 
        else:
           return  img, target_expression
           
         
    def __len__(self):
        return len(self.imgList)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list    

if __name__=='__main__':
   
   #filelist = default_reader_affectnet('../data/Affectnetmetadata/training.csv')

   #../data/Affwild2/Annotations/annotations.pkl'
   rootfolder = '../data/Affwild2/'
   transform = transforms.Compose([

                              transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]),])
   #dataset = ImageList(rootfolder, '../data/Affwild2/Annotations/test_set.pkl', 'Test',transform)
   dataset = ImageList(rootfolder, '../data/Affwild2/Annotations/annotations.pkl', 'Train',transform)
   
   fdi = iter(dataset)
   img_list = []
   target_list = []
   for i, data in enumerate(fdi):
       if i < 20:
          print(i, len(data), data[0].size(), data[1])
          #img_list.append(data[1].split('/')[-1])
          continue
       else:
          break
   img_list.sort()
   print(len(img_list), img_list)


    

