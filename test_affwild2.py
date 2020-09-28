'''
Aum Sri Sai Ram

Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 28-09-2020
Email: darshangera@sssihl.edu.in

Purpose: generate predictions on test set of Aff-Wild2

Requirements:  Create a folder ExprChallenge_predictions to store predictions of each video. It first stores predictions for all videos in a single file test_predictions.csv 
and then generate file for each video separately. 
'''
# External Libraries
import argparse
import os,sys,shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import math
import sklearn.metrics as sm
import glob
from PIL import Image
import util
#dataset class and model 

import scipy.io as sio
import numpy as np
import pdb
from statistics import mean 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from models.attentionnet import AttentionBranch, RegionBranch, AttentionLoss, count_parameters
from models.resnet import resnet50
from dataset.affectwild2_dataset import ImageList
from dataset.sampler import ImbalancedDatasetSampler
from models.losses import *

#######################################################################################################################################
# Training settings
parser = argparse.ArgumentParser(description='AffectnetWild2 expression recognition')

# DATA

parser.add_argument('--root_path', type=str, default='../data/Affwild2/',
                    help='path to root path of images')

parser.add_argument('--database', type=str, default='affectwild2',
                    help='Which Database for train. (flatcam, ferplus, affectnet)')

parser.add_argument('--metafile', type=str, default = '../data/Affwild2/Annotations/test_set.pkl',
                    help='path to training list')
'''
parser.add_argument('--test_list', type=str, default = '../data/Affwild2/Annotations/test_file.txt',
                    help='path to test list')
'''
parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=384, type=int, metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',  help='momentum')

parser.add_argument('--weight-decay', '--wd', default= 1e-3, type=float,  metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=100, type=int,metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='checkpoints_affwild2_pretrainedaff_expw_train_valid_both_again/model_best.pth.tar', type=str, metavar='PATH',   help='path to latest checkpoint (default: none)')
#parser.add_argument('--resume', default='checkpoints_affwild2_pretrainedaff_expw/16_checkpoint.pth.tar', type=str, metavar='PATH',   help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--predict_test', default=1, type=int,   help='predict score on test set(default=0)')

parser.add_argument('--model_dir','-m', default='checkpoints_affwild2', type=str)

parser.add_argument('--imagesize', type=int, default = 224, help='image size (default: 224)')

parser.add_argument('--num_classes', type=int, default=7, help='number of expressions(class)')

parser.add_argument('--num_attentive_regions', type=int, default=25, help='number of non-overlapping patches(default:25)')

parser.add_argument('--num_regions', type=int, default=4, help='number of non-overlapping patches(default:4)')

parser.add_argument('--train_rule', default='Resample', type=str, help='data sampling strategy for train loader:Resample, DRW,Reweight, None')

parser.add_argument('--loss_type', default="CE", type=str, help='loss type:Focal, CE')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser.add_argument('--workers', type=int, default = 8,
                    help='how many workers to load data')


args = parser.parse_args()


#######################################################################################################################################
def main():
    #Print args
    global args, best_prec1
    args = parser.parse_args()
    print('\n\t\t\t\t Aum Sri Sai Ram\nFER Test on AffectWild2 \n\n')
    print(args)
    print('\nimg_dir: ', args.root_path)
   
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    imagesize = args.imagesize

    best_expr_f1 = 0
    final_cm = 0
    final_mcm = 0
    best_prec1 = 0

    test_transform = transforms.Compose([
            transforms.Resize((args.imagesize,args.imagesize)),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
    
    # prepare model
    basemodel = resnet50(pretrained = False)
    attention_model = AttentionBranch(inputdim = 512, num_regions = args.num_attentive_regions, num_classes = args.num_classes)
    region_model = RegionBranch(inputdim = 1024, num_regions = args.num_regions, num_classes = args.num_classes)
    
    basemodel = torch.nn.DataParallel(basemodel).to(device)
    attention_model = torch.nn.DataParallel(attention_model).to(device)
    region_model = torch.nn.DataParallel(region_model).to(device)
    
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            basemodel.load_state_dict(checkpoint['base_state_dict'])
            attention_model.load_state_dict(checkpoint['attention_state_dict'])
            region_model.load_state_dict(checkpoint['region_state_dict'])            
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.predict_test:
       print('\n Test Mode:')
       test_dataset = ImageList(root=args.root_path,fileList='../data/Affwild2/Annotations/test_set.pkl',train_mode = 'Test', transform = test_transform)
       test_loader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=8)
       print('\n length of AffectWild2 test Database: ' + str(len(test_loader.dataset)))
       test(test_loader, basemodel, attention_model, region_model)
       create_test_output() 
    print('Sairam. Exiting. Bye.')
       

def statistic(target, predict):
    precision = sm.precision_score(target, predict, average="macro", zero_division=1)
    recall = sm.recall_score(target, predict, average="macro", zero_division=1)
    F1_score = sm.f1_score(target, predict, average="macro", zero_division=1)
    return precision, recall, F1_score


# Labels ['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise']:0,1,2,3,4,5,6  

def switch_expression(expression_argument): #convert back from Affecnet labels to Affwild2 labels
    switcher = {
          0:0,
          1:4,
          2:5,
          3:6 ,
          4:3, 5:2, 6:1,
    }
    return switcher.get(expression_argument, 0) #default neutral expression

def test(val_loader,  basemodel, attention_model, region_model):
   
    mode =  'Testing'
    basemodel.eval()
    attention_model.eval()
    region_model.eval()
    end = time.time()
    cm = 0 
   
    preds = np.empty(0, dtype=int)
    filenames = []#np.empty('',dtype=str)
    with torch.no_grad():         
        for i, (input, imgPath) in enumerate(val_loader):        
            print(i)            
            input = input.to(device) 

            attention_branch_feat, region_branch_feat = basemodel(input)
            local_features_list, global_features, attention_preds = attention_model(attention_branch_feat)
            region_preds = region_model(region_branch_feat)    
                
            all_predictions = torch.cat([attention_preds.unsqueeze(2), region_preds], dim=2)
            avg_predictions = torch.mean(all_predictions, dim=2)
            #print(avg_predictions.size())
            #preds.append(avg_predictions)
            _, pred = torch.max(avg_predictions,dim=1)
            
            preds = np.concatenate((preds,pred.cpu().numpy()))
            filenames = filenames + list(imgPath)
            #filenames.append(imgPath)
            #print(pred, imgPath)
    
    
    #print( len(filenames),filenames[:2])#, np.array(filenames).size, preds.size )
    table = np.array([[d.replace("'",""), switch_expression(int(c))] for d, c in zip(filenames, preds)])
    #print(table)
    np.savetxt('ExprChallenge_predictions/test_predictions.csv', table, delimiter='\n', fmt="%50s,%s")
 
def create_test_output(filename='ExprChallenge_predictions/test_predictions.csv'):
    d = dict()
    with open(filename,'r') as fp:
         lines = fp.readlines()#.sort()
         lines.sort()
         lines = [line.strip().split('/')[1:3] for line in lines]
         print('\ntotal: ', len(lines), lines[:2])#,lines[-20:])
          
         for line in lines:
             key,value = line[0], line[1]
             video_file = 'ExprChallenge_predictions/'+key+'.txt'
             if not os.path.exists(video_file):
                
                f = open(video_file,'w')
                f.write('Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise')
             else:
                
                f = open(video_file,'a') 
                 name, emotion = value.replace("'","").split(',')#v.replace("'","").split(',')[0], v.replace("'","").split(',')[1]
             #print(key,value, name, emotion)
             label  = emotion#switch_expression(emotion) #only here because emotion name is written
             
             f.write('\n'+ str(label))
             #print('\n'+name+' '+str(label))
             f.close()
             

    print('\nTest out created.')

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res





if __name__ == "__main__":
    #check_all_frames_predicted()
    main()
    print("Process has finished!")
   
     
    




    

