width  = 512
height = 512
out = cv2.VideoWriter('/content/gdrive/MyDrive/output.avi', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=30, frameSize=(width, height), apiPreference=0)
for images, targets, image_ids in data_loader_test:
  images = torch.stack(images).to(device)
  outputs = model(images)
  for i in range(len(outputs)):
    boxes,scores,labels = filterBoxes(outputs[i],0.2,0.4)
    print("image_ids: ",image_ids[i])
    print("boxes:",boxes)
    print("scores:",scores)
    print("labels:",labels)
    # Preprocessing
    image = cv2.imread(image_ids[i])
    image = cv2.resize(image,(width,height))
    
    idx = scores >= 0.5
    boxes = boxes[idx]
    scores = scores[idx]
    labels = labels[idx]
    
    colors = {1:(0,255,0), 2:(0,255,255), 3:(0,0,255)}
    
    for box,label in zip(boxes,labels):
      image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), colors[label], 2)

    #cv2_imshow(image)
    #cv2.waitKey(0)
    out.write(image)
out.release()