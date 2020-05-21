
import pytesseract
import time
import cv2
import argparse
import numpy as np
import os
import glob2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to input image')
ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
args = ap.parse_args()

# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) 
#all_files_test=[]
#for ext in ["*.jpg"]:
#    images_test = glob2.glob(os.path.join(args.folder+'/', ext))
#    all_files_test+=images_test 
image = cv2.imread(args.image)
#d=318
#for img in all_files_test:


Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)

outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

# Thực hiện xác định bằng HOG và SVM
start = time.time()

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
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])
         
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

for i in indices:
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
bottom=y
top=y+h
if bottom<0:
    bottom=0
    top=y+h
    if top >image.shape[0]:
        top=image.shape[0]
bottom1=x
top1=x+w
if bottom1<0:
    bottom1=0
    top1=x+w
    if top1 >image.shape[1]:
        top1=image.shape[1]
        
img_crop=image[bottom:top,bottom1:top1]
def secondCrop(img):
    gray=cv2.cvtColor(img_crop,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,0)
    contours,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if(len(areas)!=0):
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        secondCrop = img[y:y+h,x:x+w]
    else: 
        secondCrop = img
    return secondCrop
end = time.time()
#scrop=secondCrop(img_crop)
#cv2.imshow('second crop',scrop)
# Chuyen doi anh bien so
#img_crop = cv2.convertScaleAbs(img_crop, alpha=(255.0))
# Chuyen anh bien so ve gray
img_crop1=cv2.resize(img_crop,(img_crop.shape[1]+30,img_crop.shape[0]+20),interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img_crop1, cv2.COLOR_BGR2GRAY)
#hist=cv2.equalizeHist(gray)

#cv2.imshow("Anh bien so sau chuyen xam", gray)

# Ap dung threshold de phan tach so va nen
binary = cv2.threshold(gray, 127, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#cv2.imshow("Anh bien so sau threshold", binary)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # Nhan dien bien so. Cau hinh --psm 7 la de nhan dien 1 line only
text = pytesseract.image_to_string(img_crop, lang="eng", config="--psm 7")

    # Viet bien so len anh
cv2.putText(image,fine_tune(text),(50,50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 255), lineType=cv2.LINE_AA)

#python YOLO.py -i testdata/m477_0.jpg -cl obj.names -w yolov3-1c-1000-max-steps_2000.weights -c testyolov3-1c-1000-max-steps.cfg
cv2.imshow('output',image)
d=0
cv2.imwrite("result/output_{}.jpg".format(d), image)
    #d+=1

#end = time.time()
print("YOLO Execution time: " + str(end-start))


cv2.waitKey()
cv2.destroyAllWindows()



