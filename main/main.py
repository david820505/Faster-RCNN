# Import dependencies
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch
from torch.autograd import Variable
import torchvision
import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from engine import train_one_epoch, evaluate
import utils

DAY_TRAIN_PATH = 'gdrive/MyDrive/proj/Annotations/Annotations/dayTrain/'
NIGHT_TRAIN_PATH = 'gdrive/MyDrive/proj/Annotations/Annotations/nightTrain/'
DATA_PATH = 'gdrive/MyDrive/proj'
label_to_index = {'go':1,'warning':2,'stop':3}
idx_to_label = {v:k for k,v in label_to_index.items()}

def changeFilename(x):
  filename = x.Filename
  isNight = x.isNight
    
  splitted = filename.split('/')
  clipName = splitted[-1].split('--')[0]
  if isNight:
    return os.path.join(DATA_PATH,f'nightTrain/nightTrain/{clipName}/frames/{splitted[-1]}')
  else:
    return os.path.join(DATA_PATH,f'dayTrain/dayTrain/{clipName}/frames/{splitted[-1]}')


def changeAnnotation(x):
  if 'go' in x['Annotation tag']:
    return label_to_index['go']
  elif 'warning' in x['Annotation tag']:
    return label_to_index['warning']
  elif 'stop' in x['Annotation tag']:
    return label_to_index['stop']


#function to read data from csv
def csvToDf_test(path="gdrive/MyDrive/proj/Annotations/Annotations"):
  ds_path1 = os.path.join(path, "daySequence1", "frameAnnotationsBOX.csv")
  ds_path2 = os.path.join(path, "daySequence2","frameAnnotationsBOX.csv")

  df_1 = pd.read_csv(ds_path1, sep=";") 
  df_2 = pd.read_csv(ds_path2, sep=";")
  ds = df_1.append(df_2)

  ds = df_process(ds)

  return ds

def csvToDf_train(path="gdrive/MyDrive/proj/Annotations/Annotations/dayTrain"):
  #readcsv
  day_clip_path = path
  
  data = []
  for clip_name in tqdm(sorted(os.listdir(day_clip_path))):
    if 'dayClip' not in clip_name:
      continue
    df = pd.read_csv(os.path.join(day_clip_path, clip_name,'frameAnnotationsBOX.csv'), sep=";")
    data.append(df)
  #Build df
  df = pd.concat(data,axis = 0)

  df = df_process(df)

  return df

def df_process(df):
  df['isNight'] = 0

  # Droppin duplicate columns & "Origin file" as we don't need it
  df = df.drop(['Origin file','Origin track frame number','Origin track'],axis=1)
  
  # # Apply changeFilename
  df['Filename'] = df.apply(changeFilename,axis=1)
  
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
        # load all image paths
        day_df = csvToDf_train(root)
        self.imgs = imgsPath(day_df)
        self.df = day_df

    def __len__(self) -> int:
        #print("imgs shape: ", self.imgs.shape[0])
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        # load images
        #img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")

        # get bounding box coordinates for each images
        records = self.df[self.df.image_id == img_path]
        
        boxes = records[['x_min','y_min','x_max','y_max']].values
        boxes = torch.as_tensor(boxes,dtype=torch.float32)
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        labels = torch.as_tensor(records.label.values, dtype=torch.int64)
        
        iscrowd = torch.zeros_like(labels, dtype=torch.int64)

        #print("Path: ", img_path, "labels:", labels, "boxes: ", boxes)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

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

#======================================================================================================================================
#     Main
#======================================================================================================================================
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# split the dataset in train and test set
dataset = TrafficLightDataset(DAY_TRAIN_PATH, get_transform(train=True))

indices = torch.randperm(len(dataset)).tolist()
dataset_train = torch.utils.data.Subset(dataset, indices[:-2000])
dataset_test = torch.utils.data.Subset(dataset, indices[-2000:])

data_loader = torch.utils.data.DataLoader( dataset_train, batch_size=2, shuffle=True, num_workers=4, collate_fn = utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader( dataset_test, batch_size=2, shuffle=True, num_workers=4, collate_fn = utils.collate_fn)

model = FasterRCNN_model_setting()
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# let's train it for 10 epochs
num_epochs = 10
for epoch in range(num_epochs):
  # train for one epoch, printing every 10 iterations
  train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
  # update the learning rate
  lr_scheduler.step()
  # evaluate on the test dataset
  evaluate(model, data_loader_test, device=device)

print("We made it!")
