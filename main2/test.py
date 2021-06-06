from torchvision.ops import nms
from torchvision import transforms as T
import albumentations as A
import cv2
from google.colab.patches import cv2_imshow
'''
for iou_type, coco_eval in result.coco_eval.items():
    print(coco_eval._gts)#ground truth
    print(coco_eval._dts)#detection
'''

preprocess = A.Compose([
        A.Resize(height=512, width=512, p=1),
        ToTensorV2(p=1.0)
    ])
video = cv2.VideoCapture("/content/gdrive/MyDrive/proj/daySequence1.avi")
width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
out = cv2.VideoWriter('/content/gdrive/MyDrive/output.avi', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=30, frameSize=(width, height), apiPreference=0)

while(True):
    ret, input = video.read()
    if input is None:
      break
    image = input.copy()
    #print("[Input]", type(input), input)
    input = preprocess(image = input)["image"]
    #print(input)
    input = input.unsqueeze_(0)
    input = input.type(torch.cuda.FloatTensor)

    test_result = model(input)
    print("=============================================")
    print("test_result: ", test_result, len(test_result))
    boxes = test_result[0]['boxes'].type(torch.cuda.FloatTensor)
    scores = test_result[0]['scores'].type(torch.cuda.FloatTensor)
    labels = test_result[0]['labels'].type(torch.cuda.FloatTensor)
    print("1boxes : ", boxes)
    print("1scores: ", scores)
    print("1labels: ", labels)

    idx = nms(boxes,scores,0.3) # get IoU over 0.3
    boxes = boxes[idx]
    scores = scores[idx]
    labels = labels[idx]
    print("2boxes : ", boxes)
    print("2scores: ", scores)
    print("2labels: ", labels)

    boxes = boxes.data.cpu().numpy().astype(np.int32)
    scores = scores.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    print("3boxes : ", boxes)
    print("3scores: ", scores)
    print("3labels: ", labels)

    idx = scores >= 0.2 # make sure the score threshold
    boxes = boxes[idx]
    scores = scores[idx]
    labels = labels[idx]
    print("4boxes : ", boxes)
    print("4scores: ", scores)
    print("4labels: ", labels)

    colors = {1:(0,255,0), 2:(255,255,0), 3:(255,0,0)}

    for box,label in zip(boxes,labels):
        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), colors[label], 2)

    #cv2_imshow(image)
    #cv2.waitKey(0)
    out.write(image)

out.release()