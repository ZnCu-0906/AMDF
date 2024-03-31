import torch.nn.functional as F
import torch
import numpy as np

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class TotalLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,loss1,loss2,loss3,w1,w2):
        loss = loss1*w2+(loss2 *w1 + loss3 * (1-w1))*(1-w2)
        
        return loss

class caculate_MAP(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,output1, output2,output3, output4,results,signs,label,W):
        distance_day = F.pairwise_distance(output1, output2, keepdim=True)
        distance_night = F.pairwise_distance(output3, output4, keepdim=True)
        distance_numpy = (distance_day.cpu().data.numpy().T) *W \
                                 +(distance_night.cpu().data.numpy().T) *(1-W)
        results = np.concatenate((results, distance_numpy.reshape(-1)))
        label_numpy = label.cpu().data.numpy().T
        signs = np.concatenate((signs, label_numpy.reshape(-1)))
        return results,signs
    