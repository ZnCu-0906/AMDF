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
from nets.MarginLoss import MarginLoss
from torch.nn import CrossEntropyLoss
from sklearn.metrics import average_precision_score


train_batch_size = 16       
train_number_epochs = 20 
model_name = 'your model name'
w1 = 0.5                    
w2 = 0.1                    
b = 0.5
folder_dataset_train = datasets.ImageFolder("your dataset path")
folder_dataset_train_night = "your dataset path"
folder_dataset_test = datasets.ImageFolder("your dataset path")
folder_dataset_test_night = "your dataset path"

transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
                                
stam_dataset_train = STAMDataset(image_folder_dataset=folder_dataset_train,image_folder_dataset_night=folder_dataset_train_night,transform=transform,
                                        should_invert=False)

stam_dataset_test = STAMDataset(image_folder_dataset=folder_dataset_test,image_folder_dataset_night=folder_dataset_test_night,transform=transform,
                                        should_invert=False)
                                        

train_dataloader = DataLoader(stam_dataset_train,
                              shuffle=False,
                              batch_size=train_batch_size)

test_dataloader = DataLoader(stam_dataset_test,
                              shuffle=False,
                              batch_size=train_batch_size)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
head = MarginLoss(embedding_size=5000, classnum=2500,s=64., m=0.7).to(device)
net = STAM().cuda()  


optimizer = torch.optim.SGD(net.parameters(),lr=0.002,momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)
criterion_dis = ContrastiveLoss()
criterion_angle = nn.CrossEntropyLoss()
loss_total = TotalLoss()
cal_map = caculate_MAP()
model_weight = None
base_loss = 10000


for epoch in range(0,train_number_epochs):   
    optimizer.zero_grad()
    if epoch > 0:
       scheduler.step()
    net.train()
    running_loss = 0.0
    res = []
    sign = []
    for i,data in enumerate(train_dataloader):
        img1, img2, img3,img4, label,y1,y2 = data
        img1, img2, img3,img4, label,y1,y2 = \
            img1.cuda(), img2.cuda(),img3.cuda(),img4.cuda(),label.cuda(),y1.cuda(),y2.cuda()
        optimizer.zero_grad()
        output1,output2,output3,output4 = net(img1,img2,img3,img4)
        loss_dis_day = criterion_dis(output1, output2, label)
        loss_dis_night =criterion_dis(output3, output4, label)
        output1_1 = head(output1, y1, b)
        output1_2 = head(output2, y2, b)
        output1_3 = head(output3, y1, b)
        output1_4 = head(output4, y2, b)
        loss_anle = criterion_angle(output1_1, y1)+criterion_angle(output1_2, y2)+criterion_angle(output1_3, y1)+criterion_angle(output1_4, y2)
        LossT = loss_total(loss_anle,loss_dis_day,loss_dis_night,w1,w2)
        LossT.backward()
        optimizer.step()
        running_loss += LossT.item() * label.size(0)
        res,sign = cal_map(output1,output2,output3,output4,res,sign,label,w1)
        torch.cuda.empty_cache()
        average_precision = average_precision_score(sign, res)
    print("==================train==================")
    print("Epoch number {}".format(epoch))
    print("average_precision: {}\n".format(average_precision))


    net.eval()
    running_loss = 0.0
    res = []
    sign = []
    
    for i,data in enumerate(test_dataloader):
        img1, img2, img3,img4, label,y1,y2 = data
        img1, img2, img3,img4, label,y1,y2 = \
            img1.cuda(), img2.cuda(),img3.cuda(),img4.cuda(),label.cuda(),y1.cuda(),y2.cuda()
        optimizer.zero_grad()
        output1,output2,output3,output4 = net(img1,img2,img3,img4)
        loss_dis_day = criterion_dis(output1, output2, label)
        loss_dis_night =criterion_dis(output3, output4, label)
 
        output1_1 = head(output1, y1, b)
        output1_2 = head(output2, y2, b)
        output1_3 = head(output3, y1, b)
        output1_4 = head(output4, y2, b)
        loss_anle = criterion_angle(output1_1, y1)+criterion_angle(output1_2, y2)+criterion_angle(output1_3, y1)+criterion_angle(output1_4, y2)

        LossT = loss_total(loss_anle,loss_dis_day,loss_dis_night,w1,w2)
        running_loss += LossT.item() * label.size(0)
        epoch_loss = running_loss / len(stam_dataset_test)
        if epoch_loss < base_loss:
            base_loss = epoch_loss
            model_weight = copy.deepcopy(net.state_dict())

        res,sign = cal_map(output1,output2,output3,output4,res,sign,label,w1)
        torch.cuda.empty_cache()
        average_precision = average_precision_score(sign, res)
    print("==================test==================")
    print("average_precision: {}\n".format(average_precision))

print('Base loss: {:4f}'.format(base_loss))
net.load_state_dict(model_weight)
torch.save(net.state_dict(),"model_res/"+"{}.pkl".format(model_name))