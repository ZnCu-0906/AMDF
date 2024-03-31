from copy import deepcopy
from nets.build import build_model
import torch.nn as nn
import torch

class STAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin_net=build_model()
    
    def preprocessing(self,x):
        output1 = self.swin_net(x)
        norm = torch.norm(output1,2,1,True)
        output = torch.div(output1,norm)
        return output
   

    def forward(self, input_day1, input_day2, input_night1, input_night2):
        out_day1 = self.preprocessing(input_day1)
        out_day2 = self.preprocessing(input_day2)
       
        out_night1 = self.preprocessing(input_night1)
        out_night2 = self.preprocessing(input_night2)

        return out_day1, out_day2, out_night1, out_night2
    