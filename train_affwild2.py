'''
Aum Sri Sai Ram

Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 28-09-2020
Email: darshangera@sssihl.edu.in

Purpose: Perform training on Aff-Wild2
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

parser.add_argument('--metafile', type=str, default = '../data/Affwild2/Annotations/annotations.pkl',
                    help='path to training list')

parser.add_argument('--test_list', type=str, default = '../data/Affwild2/Annotations/test_file.txt',
                    help='path to test list')

parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('-b_t', '--batch-size_t', default=64, type=int, metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',  help='momentum')

parser.add_argument('--weight-decay', '--wd', default= 1e-3, type=float,  metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=100, type=int,metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',   help='path to latest checkpoint (default: none)')

parser.add_argument('--pretrained', default='pretrainedmodels/vgg_msceleb_resnet50_ft_weight.pkl', type=str, metavar='PATH', 
                    help='path to pretrained FR Model (default: none)')

parser.add_argument('-e', '--predict_test', default=0, type=int,   help='predict score on test set(default=0)')

parser.add_argument('--model_dir','-m', default='checkpoints', type=str, help='checkpoints folder')

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
    print('\n\t\t\t\t Aum Sri Sai Ram\nFER on AffectWild2 using Local and global Attention along with region branch (non-overlapping patches)\n\n')
    print(args)
    print('\nimg_dir: ', args.root_path)
    print('\ntrain rule: ',args.train_rule, ' and loss type: ', args.loss_type, '\n')
    print('\n lr is : ', args.lr)

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    imagesize = args.imagesize

    best_expr_f1 = 0
    final_cm = 0
    final_mcm = 0
    best_prec1 = 0


    train_transform = transforms.Compose([          
            transforms.RandomHorizontalFlip(p=0.5),           
            transforms.ColorJitter(brightness=0.4, contrast = 0.3, saturation = 0.25, hue = 0.05),            
            transforms.Resize((args.imagesize,args.imagesize)),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])

    
    valid_transform = transforms.Compose([
            transforms.Resize((args.imagesize,args.imagesize)),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])

    val_data = ImageList(root=args.root_path, fileList = args.metafile, train_mode='Validation',
                  transform=valid_transform)   
    
    val_loader = torch.utils.data.DataLoader(val_data, args.batch_size, shuffle=False, num_workers=8)

    train_dataset = ImageList(root=args.root_path, fileList = args.metafile,train_mode='Train',
                  transform=train_transform)


    cls_num_list = train_dataset.get_cls_num_list()
    print('\nTrain cls num list:', cls_num_list)


    if args.train_rule == 'None':
       train_sampler = None  
       per_cls_weights = None 
    elif args.train_rule == 'Resample':
       train_sampler = ImbalancedDatasetSampler(train_dataset)
       per_cls_weights = None
    elif args.train_rule == 'Reweight':
       train_sampler = None
       beta = 0.9999                 #0:normal weighting
       effective_num = 1.0 - np.power(beta, cls_num_list)
       per_cls_weights = (1.0 - beta) / np.array(effective_num)
       per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
       per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    
    if args.loss_type == 'CE':
       criterion = nn.CrossEntropyLoss(weight=per_cls_weights).to(device)
    elif args.loss_type == 'Focal':
       criterion = FocalLoss(weight=per_cls_weights, gamma=2).to(device)
    else:
       warnings.warn('Loss type is not listed')
       return
    

        
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler)    
    
    print('\nlength of AffectWild2 train Database: ' + str(len(train_dataset)))
    print('\nlength of AffectWild2 valid Database: ' + str(len(val_loader.dataset)))

    # prepare model
    basemodel = resnet50(pretrained = False)
    attention_model = AttentionBranch(inputdim = 512, num_regions = args.num_attentive_regions, num_classes = args.num_classes)
    region_model = RegionBranch(inputdim = 1024, num_regions = args.num_regions, num_classes = args.num_classes)
    
    basemodel = torch.nn.DataParallel(basemodel).to(device)
    attention_model = torch.nn.DataParallel(attention_model).to(device)
    region_model = torch.nn.DataParallel(region_model).to(device)
    
    print('\nNumber of parameters:')
    print('Base Model: {}, Attention Branch:{}, Region Branch:{} and Total: {}'.format(count_parameters(basemodel),count_parameters(attention_model),  count_parameters(region_model), count_parameters(basemodel)+count_parameters(attention_model)+count_parameters(region_model)))  
    
    

    criterion1 = AttentionLoss().to(device)
    
    optimizer =  torch.optim.SGD([{"params": basemodel.parameters(), "lr": 0.0001, "momentum":args.momentum,
                                 "weight_decay":args.weight_decay}])
    
    optimizer.add_param_group({"params": attention_model.parameters(), "lr": args.lr, "momentum":args.momentum,
                                 "weight_decay":args.weight_decay})
    
    optimizer.add_param_group({"params": region_model.parameters(), "lr": args.lr, "momentum":args.momentum,
                                 "weight_decay":args.weight_decay})
  
    if args.pretrained:
        util.load_state_dict(basemodel,'pretrainedmodels/vgg_msceleb_resnet50_ft_weight.pkl')
        
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            basemodel.load_state_dict(checkpoint['base_state_dict'])
            attention_model.load_state_dict(checkpoint['attention_state_dict'])
            region_model.load_state_dict(checkpoint['region_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.predict_test:
       print('\n Test Mode:')
       test_data = TestList(root=args.root_path, fileList='../data/Affwild2/Annotations/test_file.txt', transform=valid_transform)       
       test_loader = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=False, num_workers=8)
       print('\n length of AffectWild2 test Database: ' + str(len(test_loader.dataset)))
       test(test_loader, basemodel, attention_model, region_model)

       print('Sairam. Exiting. Bye.')
       assert(False)
       

    print('\nTraining starting:\n')
    for epoch in range(args.start_epoch, args.epochs):
        
        # train for one epoch

        train(train_loader, basemodel, attention_model, region_model, criterion, criterion1, optimizer, epoch)

        adjust_learning_rate(optimizer, epoch)

        prec1, f1, cm = validate(val_loader, basemodel, attention_model, region_model, criterion, criterion1,  epoch)
        print("Epoch: {}   Validation Acc: {}, Validation f1:{} and Final score :{}".format(epoch, prec1, f1, 0.0033*prec1+0.67*f1))
        # remember best prec@1 and save checkpoint
        
        is_best = prec1 > best_prec1 and  f1 > best_expr_f1

        final_cm += cm

        best_prec1 = max(prec1.to(device).item(), best_prec1)
        best_expr_f1 = max(f1, best_expr_f1)

        if is_best:
           print(cm)
           #np.save(os.path.join("logs","CM.npy"), np.array(final_cm))

        save_checkpoint({
            'epoch': epoch + 1,            
            'base_state_dict': basemodel.state_dict(),
            'attention_state_dict': attention_model.state_dict(),
            'region_state_dict': region_model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best.item())
        
        

def train(train_loader,  basemodel, attention_model, region_model, criterion, criterion1, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top5 = AverageMeter()
    att_loss = AverageMeter()
    region_loss = AverageMeter()
    overall_loss = AverageMeter()
    region_prec = []
     
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)

        target = target.to(device)
        #print(input.size(), target.size())
        # compute output
        attention_branch_feat, region_branch_feat = basemodel(input)
        
        local_features_list, global_features, attention_preds = attention_model(attention_branch_feat)

        region_preds = region_model(region_branch_feat)

        #Attention Branch Loss: loss1
        loss1 = criterion(attention_preds, target) #attention CELoss

        #Region Branch Loss: loss2        
        for j in range(4):
            if j == 0:
               loss2 = criterion(region_preds[:,:,j], target) #region celoss loss from Ist region branch 
            else:
               loss2 += criterion(region_preds[:,:,j], target) #region celoss loss for rest 3 regions from region branch
            
        att_loss.update(loss1.item(), input.size(0))
        region_loss.update(loss2.item(), input.size(0))

        att_wt = 0.2
        loss = att_wt * loss1 + (1 - att_wt) *loss2 # weights for both branches
        overall_loss.update(loss.item(), input.size(0))
        all_predictions = torch.cat([attention_preds.unsqueeze(2), region_preds], dim=2)
        avg_predictions = torch.mean(all_predictions, dim=2)
        avg_prec = accuracy(avg_predictions,target,topk=(1,))
        
        top1.update(avg_prec[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Training Epoch: [{0}][{1}/{2}]\t'
                  #'Time  ({batch_time.avg})\t'
                  #'Data ({data_time.avg})\t'
                  'att_loss  ({att_loss.avg})\t'
                  'region_loss ({region_loss.avg})\t'
                  'overall_loss ({overall_loss.avg})\t' 
                  'Prec1  ({top1.avg}) \t'.format(
                   epoch, i, len(train_loader), 
#                   epoch, i, len(train_loader), batch_time = batch_time, data_time=data_time, 
                  att_loss=att_loss,region_loss=region_loss,overall_loss=overall_loss,  top1=top1))

def statistic(target, predict):
    precision = sm.precision_score(target, predict, average="macro", zero_division=1)
    recall = sm.recall_score(target, predict, average="macro", zero_division=1)
    F1_score = sm.f1_score(target, predict, average="macro", zero_division=1)
    return precision, recall, F1_score

def val_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)

    top1 = pred[:, 0].cpu().numpy()
    target_np = target.view(-1).cpu().numpy()

    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    
    cm = 0
    #print(target_np.shape, top1.shape)
    if top1.size == 0 and target_np.size == 0:
       cm = 0
       precision, recall, F1_score = -1, -1, -1
    else:
       cm = sm.confusion_matrix(target_np, top1, labels=range(7),normalize='all') #'true,'pred'
       precision, recall, F1_score = statistic(target_np, top1)
    return res, cm, precision, recall, F1_score

def validate(val_loader,  basemodel, attention_model, region_model, criterion, criterion1, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    att_loss = AverageMeter()
    region_loss = AverageMeter()
    overall_loss = AverageMeter()
    #region_prec = []
    mode =  'Testing'
    # switch to evaluate mode
    basemodel.eval()
    attention_model.eval()
    region_model.eval()
    end = time.time()
    cm = 0 
    f1 = AverageMeter()
    with torch.no_grad():         
        for i, (input, target) in enumerate(val_loader):        
            data_time.update(time.time() - end)
            input = input.to(device) 
            target = target.to(device)
            attention_branch_feat, region_branch_feat = basemodel(input)
            local_features_list, global_features, attention_preds = attention_model(attention_branch_feat)
            region_preds = region_model(region_branch_feat)    
            #Attention Branch Loss: loss1
            loss1 = criterion(attention_preds, target) #attention CELoss

            #Region Branch Loss: loss2        
            for j in range(4):
                if j == 0:
                   loss2 = criterion(region_preds[:,:,j], target) #region celoss loss from Ist region branch 
                else:
                   loss2 += criterion(region_preds[:,:,j], target) #region celoss loss for rest 3 regions from region branch
                
            att_loss.update(loss1.item(), input.size(0))
            region_loss.update(loss2.item(), input.size(0))

            att_wt = 0.2
            loss = att_wt * loss1 + (1 - att_wt) * loss2 # weights for both branches

            overall_loss.update(loss.item(), input.size(0))
            all_predictions = torch.cat([attention_preds.unsqueeze(2), region_preds], dim=2)
            avg_predictions = torch.mean(all_predictions, dim=2)

            avg_prec, expr_cm, precision, recall, F1_score  = val_accuracy(avg_predictions, target, topk=(1,))       
            top1.update(avg_prec[0], input.size(0))
            f1.update(F1_score, input.size(0))
            cm += expr_cm
 
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
           
            if i % 1000 == 0:
               print('{0} [{1}/{2}]\t'                 
                  'att_loss  ({att_loss.avg})\t'
                  'region_loss ({region_loss.avg})\t'
                  'Prec@1  ({top1.avg})\t' 'F1  ({f1.avg})\t'
                  .format(mode, i, len(val_loader),  att_loss=att_loss, region_loss=region_loss, top1=top1, f1=f1))
            
        print('{0} [{1}/{2}]\t'
                  #'Time {batch_time.val} ({batch_time.avg})\t'
                  'att_loss  ({att_loss.avg})\t'
                  'region_loss ({region_loss.avg})\t'
                  'overall_loss ({overall_loss.avg})\t' 
                  'Prec@1  ({top1.avg})\t' 'F1  ({f1.avg})\t'
                  .format(mode, i, len(val_loader),  att_loss=att_loss, region_loss=region_loss, overall_loss=overall_loss,  top1=top1, f1=f1))


    return top1.avg, f1.avg, cm

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    epoch_num = state['epoch']
    full_filename = os.path.join(args.model_dir, str(epoch_num)+'_'+ filename)
    full_bestname = os.path.join(args.model_dir, 'model_best.pth.tar')
    
    torch.save(state, full_filename)

    if epoch_num%1==0 and epoch_num>=0:
        torch.save(state, full_filename)
    
    if is_best:
        #torch.save(state, full_bestname)
        shutil.copyfile(full_filename, full_bestname)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed"""
    print('\n******************************\n\tAdjusted learning rate: '+str(epoch) +'\n')
    i = 0
    for param_group in optimizer.param_groups:
        if  i == 0:    
           print('\tBase old lr is: ',param_group['lr'])
           param_group['lr'] *= 0.95
           print('\tBase new lr is: ',param_group['lr'])
        else :
           print('\tBranches old lr is: ',param_group['lr'])
           param_group['lr'] *= 0.95
           print('\tBranches new lr is: ',param_group['lr'])
        i += 1   
    print('******************************')



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
    
    main()
    print("Process has finished!")
   
     
    




    

