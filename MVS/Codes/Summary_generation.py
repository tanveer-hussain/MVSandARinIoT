import cv2
import numpy as np
from sklearn.metrics import mutual_info_score
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename

start_time = time.time()
detector = cv2.ORB_create()

Tk().withdraw()
path1 = askopenfilename(initialdir = "E:\Research\Datasets\MVS",
                        filetypes = (("Image File" , "*.avi"),("All Files","*.*"),("Image File" , "*.mp4")),
                        title = "Please choose first video")

classes = 'Models/object.names'
weights = r'D:\My Research\Video Summarization\MVS\4. ICCV\Codes\MVS\Models/yolov3.weights'
config = 'Models/yolov3.cfg'

with open(classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file

net = cv2.dnn.readNet(weights, config)
def get_output_layers(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def tinu(image):

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    # create input blob
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    # apply non-max suppression

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

#    # go through the detections remaining
#    # after nms and draw bounding box
#
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]


        #draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    return class_ids


def main():
    counter = 0
    video1 = cv2.VideoCapture(path1)
    status_v1, frame1_v1 = video1.read()

    total_frames = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
    while(counter < total_frames):
        status_v1 , frame2_v1 = video1.read()
        counter = counter + 1
        #print (counter)
        if counter%15==0:
            if status_v1 is True:
                class_ids = tinu(frame1_v1)

                persons = class_ids.count(0)

                if persons >= 1:
                    #cv2.imshow('pic',frame1_v1)
                    kp1 = detector.detect(frame1_v1, None)
                    kp1 , des1 = detector.compute(frame1_v1, kp1)
                    des1 = np.array(des1)
                    des1 = cv2.resize(des1, (500,32), interpolation = cv2.INTER_AREA)
                    des1 = np.reshape(des1, (16000))

                    kp2 = detector.detect(frame2_v1, None)
                    kp2 , des2 = detector.compute(frame2_v1, kp2)
                    des2 = np.array(des2)
                    des2 = cv2.resize(des2, (500,32), interpolation = cv2.INTER_AREA)
                    des2 = np.reshape(des2, (16000))


                    mi = mutual_info_score(des1,des2)
                    if mi >= 3.7:
                        name = 'Keyframes-0\Keyframe-'+str(counter)+'.jpg'
                        cv2.imwrite(name,frame1_v1)
                    print ('mutual information = ' , mi, ', persons = ', persons)

        frame1_v1 = frame2_v1
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

main()

end_time = time.time()
