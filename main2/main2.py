# -*- coding: utf-8 -*-
"""object_dection05.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_Lw31x-ChDLeshm9xVs9_4RCT9Hn_bVE
"""

from google.colab import drive
drive.mount('/content/gdrive')

# Commented out IPython magic to ensure Python compatibility.
import os
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
import pandas as pd
import torchvision
from engine import train_one_epoch, evaluate
import utils
import torchvision.transforms as transforms
import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import nms
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
# %matplotlib inline

# OpenCV
import cv2

#Helper library
import albumentations as A
from albumentations.pytorch import ToTensorV2


#Must install this library before every time you run this program
!pip install -U git+https://github.com/albumentations-team/albumentations




DAY_TRAIN_PATH = 'gdrive/MyDrive/proj/Annotations/Annotations/dayTrain/'
NIGHT_TRAIN_PATH = 'gdrive/MyDrive/proj/Annotations/Annotations/nightTrain/'

TRAIN_PATH = 'gdrive/MyDrive/proj/Annotations/Annotations/'
DATA_PATH = 'gdrive/MyDrive/proj'
label_to_index = {'go':1,'warning':2,'stop':3}
idx_to_label = {v:k for k,v in label_to_index.items()}

def changeFilename_train(x):
  filename = x.Filename
  isNight = x.isNight
    
  splitted = filename.split('/')
  clipName = splitted[-1].split('--')[0]

  if isNight:
    return os.path.join(DATA_PATH,f'nightTrain/nightTrain/{clipName}/frames/{splitted[-1]}')
  else:
    return os.path.join(DATA_PATH,f'dayTrain/dayTrain/{clipName}/frames/{splitted[-1]}')    

def changeFilename_test(x, index=1):
  filename = x.Filename
  isNight = x.isNight
  splitted = filename.split('/')

  if isNight:
    return os.path.join(DATA_PATH,f'nightSequence{index}/nightSequence{index}/frames/{splitted[-1]}')
  else:
    return os.path.join(DATA_PATH,f'daySequence{index}/daySequence{index}/frames/{splitted[-1]}')

def changeAnnotation(x):
  if 'go' in x['Annotation tag']:
    return label_to_index['go']
  elif 'warning' in x['Annotation tag']:
    return label_to_index['warning']
  elif 'stop' in x['Annotation tag']:
    return label_to_index['stop']

#function to read data from csv
def csvToDf_test(path="gdrive/MyDrive/proj/Annotations/Annotations"):

  #process day_seqence
  ds_path1 = os.path.join(path, "daySequence1", "frameAnnotationsBOX.csv")
  ds_path2 = os.path.join(path, "daySequence2","frameAnnotationsBOX.csv")

  df_1 = pd.read_csv(ds_path1, sep=";") 
  df_1 = df_process_test(df_1, 0, index= 1)

  df_2 = pd.read_csv(ds_path2, sep=";")
  df_2 = df_process_test(df_2, 0, index= 2)

  ds = pd.concat([df_1, df_2], axis = 0)
  
  #process night_seqence
  ns_path1 = os.path.join(path, "nightSequence1", "frameAnnotationsBOX.csv")
  ns_path2 = os.path.join(path, "nightSequence2","frameAnnotationsBOX.csv")
  
  df_1 = pd.read_csv(ns_path1, sep=";") 
  df_1 = df_process_test(df_1, 1, index= 1)

  df_2 = pd.read_csv(ns_path2, sep=";")
  df_2 = df_process_test(df_2, 1, index= 2)

  #concat day_seqence and night_seqence
  ns = pd.concat([df_1, df_2], axis = 0)

  df = pd.concat([ds, ns], axis = 0)

  return df



def csvToDf_train(path="gdrive/MyDrive/proj/Annotations/Annotations"):
  #readcsv
  data = []

  day_clip_path = os.path.join(path, "dayTrain")
  for clip_name in tqdm(sorted(os.listdir(day_clip_path))):
    if 'dayClip' not in clip_name:
      continue
    df = pd.read_csv(os.path.join(day_clip_path, clip_name,'frameAnnotationsBOX.csv'), sep=";")
    data.append(df)

  #Build df
  day_df = pd.concat(data, axis = 0)
  day_df = df_process_train(day_df, 0)

  data = []
  night_clip_path = os.path.join(path, "nightTrain")
  for clip_name in tqdm(sorted(os.listdir(night_clip_path))):
    if 'nightClip' not in clip_name:
      continue
    df = pd.read_csv(os.path.join(night_clip_path, clip_name,'frameAnnotationsBOX.csv'), sep=";")
    data.append(df)
  
  #Build df
  night_df = pd.concat(data, axis = 0)
  night_df = df_process_train(night_df, 1)

  df = pd.concat([day_df, night_df], axis = 0)

  return df
  
  

#function to read data from csv
def csvToDf_test(path="gdrive/MyDrive/proj/Annotations/Annotations"):

  #process day_seqence
  ds_path1 = os.path.join(path, "daySequence1", "frameAnnotationsBOX.csv")
  ds_path2 = os.path.join(path, "daySequence2","frameAnnotationsBOX.csv")

  df_1 = pd.read_csv(ds_path1, sep=";") 
  df_1 = df_process_test(df_1, 0, index= 1)

  df_2 = pd.read_csv(ds_path2, sep=";")
  df_2 = df_process_test(df_2, 0, index= 2)

  ds = pd.concat([df_1, df_2], axis = 0)
  
  #process night_seqence
  ns_path1 = os.path.join(path, "nightSequence1", "frameAnnotationsBOX.csv")
  ns_path2 = os.path.join(path, "nightSequence2","frameAnnotationsBOX.csv")
  
  df_1 = pd.read_csv(ns_path1, sep=";") 
  df_1 = df_process_test(df_1, 1, index= 1)

  df_2 = pd.read_csv(ns_path2, sep=";")
  df_2 = df_process_test(df_2, 1, index= 2)

  #concat day_seqence and night_seqence
  ns = pd.concat([df_1, df_2], axis = 0)

  df = pd.concat([ds, ns], axis = 0)

  return df





def df_process_test(df, is_night = 0, index= 1):
  df['isNight'] = is_night

  # Droppin duplicate columns & "Origin file" as we don't need it
  df = df.drop(['Origin file','Origin track frame number','Origin track'],axis=1)
  
  # Apply changeFilename
  df['Filename'] = df.apply(changeFilename_test, axis=1, **{'index': index})

  # Apply changeAnnotation
  df['Annotation tag'] = df.apply(changeAnnotation,axis = 1)

  # Changing Column Names
  df.columns = ['image_id','label','x_min','y_min','x_max','y_max','frame','isNight']

  return df


def df_process_train(df, is_night = 0, index= "train"):
  df['isNight'] = is_night

  # Droppin duplicate columns & "Origin file" as we don't need it
  df = df.drop(['Origin file','Origin track frame number','Origin track'],axis=1)
  
  # Apply changeFilename
  df['Filename'] = df.apply(changeFilename_train, axis=1)

  # Apply changeAnnotation
  df['Annotation tag'] = df.apply(changeAnnotation,axis = 1)

  # Changing Column Names
  df.columns = ['image_id','label','x_min','y_min','x_max','y_max','frame','isNight']

  return df



def label(df):
  label_list = df['label'].tolist()
  return label_list

def imgsPath(df):
  imgs_list = df['image_id'].unique()
  print("image list:", imgs_list)
  return imgs_list

def locBox(df):
  xmin = df['x_min'].tolist()
  ymin = df['y_min'].tolist()
  xmax = df['x_max'].tolist()
  ymax = df['y_max'].tolist()
  #print("Length:", len(xmin))
  return xmin,ymin,xmax,ymax

class TrafficLightDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        # load all image paths
        day_df = csvToDf_train(root)
        self.imgs = imgsPath(day_df)
        self.df = day_df


    def __getitem__(self, idx):
        # load images
        img_path = self.imgs[idx]

        # Reading Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # get bounding box coordinates for each images
        records = self.df[self.df.image_id == img_path]

        #boxes
        boxes = records[['x_min','y_min','x_max','y_max']].values
        boxes = torch.as_tensor(boxes,dtype=torch.float32)

        labels = torch.as_tensor(records.label.values, dtype=torch.int64)

        iscrowd = torch.zeros_like(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        #if self.transforms is not None:
            #img, target = self.transforms(img, target)
        if self.transforms:
          sample = {
              'image': image,
              'bboxes': target['boxes'],
              'labels': labels
          }
          sample = self.transforms(**sample)
          image = sample['image']
            
          # target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
          target['boxes'] = torch.as_tensor(sample['bboxes'],dtype=torch.float32)
          target['labels'] = torch.as_tensor(sample['labels'])  

        return image, target,img_path

    def __len__(self)->int:
        return self.imgs.shape[0]



#Tutorial transforms
#def get_transform(train):
#    transforms = []
#    transforms.append(T.ToTensor())
#    if train:
#        transforms.append(T.RandomHorizontalFlip(0.5))
#    return T.Compose(transforms)

def getTrainTransform():
    return A.Compose([
        A.Resize(height=512, width=512, p=1),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# For Validation Data
def getValTransform():
    return A.Compose([
        A.Resize(height=512, width=512, p=1),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# For Test Data
def getTestTransform():
    return A.Compose([
        A.Resize(height=512, width=512, p=1),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})



def FasterRCNN_model_setting():
  # load a model pre-trained pre-trained on COCO
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  # replace the classifier with a new one, that has
  # num_classes which is user-defined
  num_classes = 4  # 3 class (3 diff. lights) + background
  # get number of input features for the classifier
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  # replace the pre-trained head with a new one
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  # load a pre-trained model for classification and return
  # only the features
  backbone = torchvision.models.mobilenet_v2(pretrained=True).features
  # FasterRCNN needs to know the number of
  # output channels in a backbone. For mobilenet_v2, it's 1280
  # so we need to add it here
  backbone.out_channels = 1280

  # let's make the RPN generate 5 x 3 anchors per spatial
  # location, with 5 different sizes and 3 different aspect
  # ratios. We have a Tuple[Tuple[int]] because each feature
  # map could potentially have different sizes and
  # aspect ratios
  anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

  # let's define what are the feature maps that we will
  # use to perform the region of interest cropping, as well as
  # the size of the crop after rescaling.
  # if your backbone returns a Tensor, featmap_names is expected to
  # be [0]. More generally, the backbone should return an
  # OrderedDict[Tensor], and in featmap_names you can choose which
  # feature maps to use.
  roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
  return model

bs  = 8
nw = 4
ep = 0
# SGD setting
learn = 0.005
mom = 0.9
wd = 0.0005
#file name
filename = 'SGD_lr' + str(learn) + '_mom' + str(mom) + '_wd' + str(wd) + '_b' + str(bs) + '_n' + str(nw) + '_ep' + str(ep)
#filename = 'Adam_lr' + str(learn) + '_mom' + 'None' + '_wd' + str(wd) + '_b' + str(bs) + '_n' + str(nw) + '_ep' + str(ep)
print("filename: ", filename)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# split the dataset in train and test set
#dataset = TrafficLightDataset(TRAIN_PATH, get_transform(train=True))
dataset = TrafficLightDataset(TRAIN_PATH, getTrainTransform())

indices = torch.randperm(len(dataset)).tolist()

part7 = len(indices)//10*7
part3 = len(indices)-part7

dataset_train = torch.utils.data.Subset(dataset, indices[:part7+1])#part7+1
dataset_val = torch.utils.data.Subset(dataset, indices[-part3:])#-part3


data_loader = torch.utils.data.DataLoader( dataset_train, batch_size=bs, shuffle=True, num_workers=nw, collate_fn = utils.collate_fn)
data_loader_val = torch.utils.data.DataLoader( dataset_val, batch_size=bs, shuffle=True, num_workers=nw, collate_fn = utils.collate_fn)

images, targets, image_ids = next(iter(data_loader))

#Sample testt
boxes = targets[0]['boxes'].cpu().numpy().astype(np.int32)
image = images[0].permute(1,2,0).cpu().numpy()

def displayImage(image, boxes):
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box in boxes:
      cv2.rectangle(image,(box[0], box[1]),(box[2], box[3]),(220, 0, 0), 3)

    ax.set_axis_off()
    ax.imshow(image)

    plt.show()

displayImage(image,boxes)


model = FasterRCNN_model_setting()
model.to(device)

if os.path.exists("gdrive/MyDrive/proj/"+filename+".pth"):
  model_fn = torch.load("gdrive/MyDrive/proj/"+filename+".pth")
  model.load_state_dict(model_fn)

# construct an optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = learn, momentum = mom, weight_decay = wd)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# let's train it for 10 epochs
num_epochs = 10
result = 0
for epoch in range(num_epochs):
  # train for one epoch, printing every 10 iterations
  train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
  # update the learning rate
  lr_scheduler.step()
  # evaluate on the test dataset
  result = evaluate(model, data_loader_val, device=device)
  #print("I am Coco:",result)

model.eval()
images, targets, image_ids = next(iter(data_loader_val))
images = torch.stack(images).to(device)

outputs = model(images)


def filterBoxes(output,nms_th=0.3,score_threshold=0.5):
    
    boxes = output['boxes']
    scores = output['scores']
    labels = output['labels']
    
    # Non Max Supression
    mask = nms(boxes,scores,nms_th)
    
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    boxes = boxes.data.cpu().numpy().astype(np.int32)
    scores = scores.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    return boxes, scores, labels

def displayPredictions(image_id,output,nms_th=0.3,score_threshold=0.5):
    
    boxes,scores,labels = filterBoxes(output,nms_th,score_threshold)
    print("boxes:",boxes)
    print("scores:",scores)
    print("labels:",labels)
    
    # Preprocessing
    image = cv2.imread(image_id)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = cv2.resize(image,(512,512))
    image /= 255.0
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    colors = {1:(0,255,0), 2:(255,255,0), 3:(255,0,0)}
    
    for box,label in zip(boxes,labels):
        image = cv2.rectangle(image,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      colors[label], 2)

    ax.set_axis_off()
    ax.imshow(image)

    plt.show()

displayPredictions(image_ids[0],outputs[0],0.2,0.4)

