import math
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
# import torchvision
import torchvision.transforms as transforms
from torch.nn import Parameter
import torch.nn as nn
# import torch.optim.lr_scheduler as lr_scheduler

from efficientnet_pytorch import EfficientNet


## initial setting
img_h,img_w=128, 128

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

val_trans = transforms.Compose([
    #transforms.Grayscale(num_output_channels=1),
    transforms.Resize((img_h,img_w)), #288, 144
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5], std=[0.5])
    normalize
])


## build model
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, 
                 s=30.0, m=0.50, 
                 easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        
        cosine = F.linear(F.normalize(input_), F.normalize(self.weight))
        #print("cosine.size()",cosine.size())
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        #one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)
        #print(" label.view(-1, 1)", label.view(-1, 1))
        try:
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        except Exception as e:
            print(e)
            print(" label.view(-1, 1)", label.view(-1, 1))
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
    
    
class FeatureExtractModel(nn.Module):
    def __init__(self,class_num=587, feature_num=512):
        super(FeatureExtractModel, self).__init__()
        basemodel = EfficientNet.from_pretrained('efficientnet-b0',advprop=False)
    
        self.basemodel = basemodel
            
        num_ch = feature_num
            
        print("num_ch: ", num_ch)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 3 channel output for visualization
        #class_num = trainset.class_n
        self.final = ArcMarginProduct(num_ch, class_num,
                                          s=35, m=0.5, easy_margin=False)
        
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, num_ch)
        
        self.relu1 = nn.ReLU()
        
        self.batchN1= nn.BatchNorm1d(1000)
        self.batchN2= nn.BatchNorm1d(512)
        #self.batchN3= nn.BatchNorm1d(128)
        #self.batchN2= nn.BatchNorm2d(128)
        
        self.drop1 = nn.Dropout(p=0.3)
        self.drop2 = nn.Dropout(p=0.3)
  
    def extract(self, x):
        # extract features
        x = self.basemodel(x)
        x = self.batchN1(x)
        x= self.drop1(x)
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.batchN2(x)
        x = self.drop1(x)
            
        x = self.fc2(x)
        
        return x
    
    def metric_fc(self, x, label):
        out = self.final(x, label) # pass to arcface layer
        return out
    
def cosin_metric(x1, x2):
    """
    Calculate cosine similarity between two arrays
    Args:
        x1: first input array
        x2: second input array
    """
    x1=x1.cpu().detach().numpy()
    x2=x2.cpu().detach().numpy()
    return abs(1- (np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))))

def get_img_feature(a_img_p, model):
    """
    Torch Model Inference
    Args:
        a_img_p: local image path
        model: pretrained torch model
    Return:
        a_out: Inference data
    """
    a_img = Image.open(a_img_p)
    a_img = val_trans(a_img) #trans(a_img)
    a_img = torch.unsqueeze(a_img,0)
    model.eval()
    with torch.no_grad():
        a_out = model.extract(a_img.to(device))
        
        #features = model.extract_features(a_img.to(device))
        #z= nn.MaxPool2d(2)(features)
        #a_out = z.view(-1, 1280)
        
    return a_out


if __name__ == "__main__":
    # define model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device='cpu'
    print("device:", device)
    
    class_num=586
    model = FeatureExtractModel(class_num=class_num+1, feature_num=8)
    model = model.to(device)
    
    model.load_state_dict(torch.load("./models/class_feature_8_20210830_090951_92.pth", map_location=device))
    
    if False:
        #빈 값 넣어서 모델 정상인지 확인
        # test
        x_ = torch.rand([4,3,64,64]).to(device)
        label_ =  torch.tensor([1]).to(device)
        x_f = model.extract(x_)
        print("x_f",x_f.shape)
        last_out = model.metric_fc(x_f,label_)
        print("last_out:",last_out.shape)
        
    ## predict
    img_path = "./samples/p145010new_45_1.jpg"
    img_f = get_img_feature(img_path, model)
    print("img_f:",img_f)