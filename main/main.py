# Import dependencies
import numpy as np
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import cv2
import random
import os
import numpy as np
import pandas as pd
import matplotlib.patches as patches

if __name__ == "__main__":
    # dataset <=> daySequence1
    annotation_path_box = "gdrive/MyDrive/proj/Annotations/Annotations/daySequence1/frameAnnotationsBOX.csv"
    #annotation_path_bulb = "gdrive/MyDrive/proj/Annotations/Annotations/daySequence1/frameAnnotationsBULB.csv"
    frames_path = "gdrive/MyDrive/proj/daySequence1/daySequence1/frames/"
    #print(os.listdir(frames_path))
    # choose the first image
    frame_id = os.listdir(frames_path)[0]
    print("FrameID:", frame_id, "; From: ", frames_path)
    annotations = []
    with open(annotation_path_box) as fp:  
        line = fp.readline()
        line = fp.readline() # Skip header line with descriptions
        #cnt = 1
        while line:
            annotation_file_row = (line.strip()).split(";")
            annotation_file_id = annotation_file_row[0].split("/")[1]
            if annotation_file_id == frame_id:
                annotations.append(annotation_file_row)
                print("ID:", annotation_file_id, "Row Data:", annotation_file_row)
            line = fp.readline()

    # Plot the original image - refer the path from annotation_path_box
    color_space = [(0,255,0),(255,0,0),(255,255,0)] # [0]: Green    [1]: Red    [2]:Orange
    # Read the image from the path
    frame_path = os.path.join(os.path.join(frames_path, frame_id))
    img = cv2.imread(frame_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.rcParams['figure.figsize'] = [32, 16]
    plt.imshow(img)
    plt.show()
    print("Found {} annotations:".format(len(annotations)))

    # Plot the box by the annotation - refer the path from annotation_path_box
    for annotation in annotations:
        anno_class = annotation[1]
        anno_upperleft_x = int(annotation[2])
        anno_upperleft_y = int(annotation[3])
        anno_lowerright_x = int(annotation[4])
        anno_lowerright_y = int(annotation[5])
        print("[Class]", anno_class, "[Position]", anno_upperleft_x, ",", anno_upperleft_y, ",", anno_lowerright_x, ",",  anno_lowerright_y)
        if anno_class == "go" or anno_class == "goLeft" or anno_class == "goForward":
            color_class = color_space[0]
        elif anno_class == "stop" or anno_class == "stopLeft":
            color_class = color_space[1]
        elif anno_class == "warning" or anno_class == "warningLeft":
            color_class = color_space[2]
        cv2.rectangle(img, (anno_upperleft_x, anno_upperleft_y), (anno_lowerright_x, anno_lowerright_y), color_class, 2)

    plt.imshow(img)
    plt.show()