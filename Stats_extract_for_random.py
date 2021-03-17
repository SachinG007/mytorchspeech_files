import torch
import numpy as np
import ext_transforms as et
from custom_datasets import *

transform_seg=et.ExtCompose([
    #et.ExtResize( 512 ),
    #et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
    et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
    et.ExtRandomHorizontalFlip(),
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])
transform_seg2=et.ExtCompose([
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])

splits = [0.05,0.06]#,0.07,0.08,0.09,0.1]
# splits = [0.1,0.15,0.2,0.25,0.3,0.35,0.4]
train_dataset=Cityscapes(str("../")+"/cityscapes",split='train',transform=transform_seg)
area1=[]
area2=[]
for split in splits:
    sampled_indices=torch.load("randomsampledind"+str(split)+"q1.log")
    current_indices=torch.load("randomcurrentind"+str(split)+"q1.log")
    area_split_class1=0 # class 1 is 15
    area_split_class2=0 # class 2 is 16
    l1=len(sampled_indices)
    for t in range(l1):
        _,label1=train_dataset[sampled_indices[t]]
        i1=0
        j1=0
        while(i1<label1.shape[0]):
            j1=0
            while(j1<label1.shape[1]):
                if(label1[i1,j1]==15):
                    area_split_class1=area_split_class1+1
                elif(label1[i1,j1]==16):
                    area_split_class2=area_split_class2+1
                j1=j1+1
            i1=i1+1

    area1.append(area_split_class1)
    area2.append(area_split_class2)

print("Area of class "+"15"+" is "+str(area1))
print("Area of class "+"16"+" is "+str(area2))
