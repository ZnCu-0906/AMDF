import os
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import copy
from Dataset import STAMDataset
from Loss import ContrastiveLoss,TotalLoss,caculate_MAP
from Network import STAM

from torch.nn import CrossEntropyLoss
from sklearn.metrics import average_precision_score


test_batch_size = 1      
model_name = 'your model name'
W1 = 0.5                    
                
map_location = torch.device('cuda:0') 
folder_dataset_test = datasets.ImageFolder("your dataset path")
folder_dataset_test_night = "your dataset path"

transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
                                

stam_dataset_test = STAMDataset(image_folder_dataset=folder_dataset_test,image_folder_dataset_night=folder_dataset_test_night,transform=transform,
                                        should_invert=False)
                                        

test_dataloader = DataLoader(stam_dataset_test,
                              shuffle=False,
                              batch_size=test_batch_size)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = STAM().cuda()  
net.load_state_dict(torch.load("model_res/"+ '{}.pkl'.format(model_name), map_location=map_location))
signs=[]
results=[]
cal_map = caculate_MAP()

    
for i,data in enumerate(test_dataloader):
    img1, img2, img3,img4, label,y1,y2 = data
    img1, img2, img3,img4, label,y1,y2 = \
        img1.cuda(), img2.cuda(),img3.cuda(),img4.cuda(),label.cuda(),y1.cuda(),y2.cuda()

    x1 = Variable(img1)
    x2 = Variable(img2)
    x3 = Variable(img3)
    x4 = Variable(img4)
    with torch.no_grad():
        output1,output2,output3,output4 = net(x1,x2,x3,x4)
    results,signs = cal_map(output1,output2,output3,output4,results,signs,label,W1)
average_precision = average_precision_score(signs, results)
print("==================test==================")
print("average_precision: {}\n".format(average_precision))
