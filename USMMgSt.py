
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib_inline
import pandas as pd
import os 
import glob

def plotimage_withbbox(img,gt):
    yx=np.shape(img)
    
    xg = gt[0]*yx[1]
    yg = gt[1]*yx[0]
    wg = gt[2]*yx[1]
    hg = gt[3]*yx[0]
    
    xgt=int(xg-(wg/2))
    ygt=int(yg-(hg/2))
    wgt=int(wg)
    hgt=int(hg)
    cv2.rectangle(img,(xgt,ygt),(xgt+wgt,ygt+hgt),(36,255,12),5)
    cv2.putText(img, 'Groundtruth', (1200, 600), cv2.FONT_HERSHEY_SIMPLEX,4, (36,255,12), 5)
    plt.imshow(img)


def readrobot(robotname,batch):
    if batch == 1:
        pathim = robotname+"/images/test/*.png"
        pathlabel = robotname+"/labels/test/"
    elif batch == 2:
        pathim = robotname+"/images/train/*.png"
        pathlabel = robotname+"/labels/train/"
    elif batch == 3:
        pathim = robotname+"/images/val/*.png"
        pathlabel = robotname+"/labels/val/"
    labels = readlabelsgt(pathlabel)

    images = [cv2.imread(file) for file in glob.glob(pathim)]
    return labels,images




def load_image(name, path):
    img_path = path + name
    img = cv2.imread(img_path)
    return img


def plot_image(img):
    plt.imshow(img)
    plt.title(img.shape)
    
def plot_image2(img,title):
    fig = plt.imshow((img * 255).astype(np.uint32)) 

    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(title)
    # plt.savefig('here.png', bbox_inches='tight',transparent=True, pad_inches=0)
    # plt.imsave('here2.png', img)
    
def plot_grid(img_names, img_root, rows=5, cols=5):
    fig = plt.figure(figsize=(25,25))
    
    for i,name in enumerate(img_names):
        fig.add_subplot(rows,cols,i+1)
        img = load_image(name, img_root)
        plot_image(img)   
    plt.show()

# def plotimage_withbbox(img,gt):
#     xg = gt[0]*1920
#     yg = gt[1]*1080
#     wg = gt[2]*1920
#     hg = gt[3]*1080
#     xgt=int(xg-(wg/2))
#     ygt=int(yg-(hg/2))
#     wgt=int(wg)
#     hgt=int(hg)
#     cv.rectangle(img,(xgt,ygt),(xgt+wgt,ygt+hgt),(36,255,12),5)
#     cv.putText(img, 'Groundtruth', (1200, 600), cv.FONT_HERSHEY_SIMPLEX,4, (36,255,12), 5)
#     plt.imshow(img)
    
def draw_bounding_box(img, gt,annotation,annotation1,annotation2,annotation3):
    x1=int(annotation1[0])
    y1=int(annotation1[1])
    w1=int(annotation1[2])
    h1=int(annotation1[3])
    
    x2=annotation2[0]*1920
    y2=annotation2[1]*1080
    w2=annotation2[2]*1920
    h2=annotation2[3]*1080

    ww2= x2+(w2/2)
    hh2= y2+(h2/2)
    x2new = x2-(w2/2)
    y2new = y2-(h2/2)
    
    x2new= int(x2new)
    y2new = int(y2new)
    ww2 = int(ww2)
    hh2 = int(hh2)
    
    
    
    x3=annotation3[0]*1920
    y3=annotation3[1]*1080
    w3=annotation3[2]*1920
    h3=annotation3[3]*1080

    ww3= x3+(w3/2)
    hh3= y3+(h3/2)
    x3new = x3-(w3/2)
    y3new = y3-(h3/2)
    
    x3new= int(x3new)
    y3new = int(y3new)
    ww3 = int(ww3)
    hh3 = int(hh3)
    
    

    x=int(annotation[0])
    y=int(annotation[1])
    w=int(annotation[2])
    h=int(annotation[3])
    xgt=int(gt[0])
    ygt=int(gt[1])
    wgt=int(gt[2])
    hgt=int(gt[3])
    

    cv.rectangle(img,(xgt,ygt),(xgt+wgt-25,ygt+hgt-10),(36,255,12),4)
    cv.putText(img, 'Groundtruth', (1200, 600), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (36,255,12), 5)

    cv.rectangle(img,(x+20,y),(x+w-15,y+h-15),(255,0,0),3)
#     cv2.putText(img, 'MaskRCNN', (1200, 700), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,0,0), 5)
    
    cv.rectangle(img,(x1+15,y1+4),(x1+w1,y1+h1),(36,12,255),3)
#     cv2.putText(img, 'FasterRCNN', (1200,800), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (36,12,255), 5)
    
    cv.rectangle(img,(x2new+20,y2new-3),(ww2,hh2),(255,12,255),3)
#     cv2.putText(img, 'Tphyolov5', (1200, 900), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,15,255), 5)
    
    cv.rectangle(img,(x3new-1,y3new+20),(ww3,hh3),(15,255,255),3)
#     cv2.putText(img, 'Yolov5', (1200, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (15,255,255), 5)
    
def plot_gridwithboundingbox(img_names, img_root,results,annotations, rows, cols):
#     fig = plt.figure(figsize=(rows*cols,rows*cols))   
    fig = plt.figure(figsize=(25,25))
    
    for i,name in enumerate(img_names):
        number = int(''.join(filter(str.isdigit, name)))
        fig.add_subplot(rows,cols,i+1)
        img = load_image(name, img_root)
        draw_bounding_box(img, annotations[number+i]['bbox'] ,results.get("bbox")[number+i])
        D = eudist(results,annotations,i)
        title = "Eucdist:  gt & detection= "+ str(D)
        plot_image2(img,title)
    plt.show()    
    
def getfilenames(images):
    filenames=[]
    for im in range(0,len(images)):
        filenames.append(images[im].get("file_name"))
    return filenames
def eudist(results,annotations,randomnumber):
    detectionsres = results.get("bbox")[randomnumber]
    imagegt = annotations[randomnumber]['bbox']
    a0,b0 = ((detectionsres[0] + detectionsres[1]) * 0.5, (detectionsres[2] + detectionsres[3]) * 0.5)
    a1,b1 = ((imagegt[0] + imagegt[1]) * 0.5, (imagegt[2] + imagegt[3]) * 0.5)
    D = dist.euclidean((a0, b0), (a1,b1))
    return D
def alleudist(results,annotations):
    Dvec=[]
    for i in range(0,len(results.get("bbox"))):
        Dvec.append(eudist(results,annotations,i))
    return Dvec
def get_iou2(bb1, bb2):
    x_left = max(float(bb1[0]),float(bb2[0]))
    y_top = max(float(bb1[1]),float(bb2[1]))
    x_right = min (float(bb1[0])+float(bb1[2]),float(bb2[0])+float(bb2[2]))
    y_bottom  = min (float(bb1[1])+float(bb1[3]),float(bb2[1])+float(bb2[3]))

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    w = float(bb1[2])
    h = float(bb1[3])
    wgt =float(bb2[2])
    hgt = float(bb2[3])
    bb1_area = w*h
    bb2_area = wgt*hgt
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou

def get_iou(bb1, bb2):
    x_left = max(bb1[0],bb2[0])
    y_top = max(bb1[1],bb2[1])
    x_right = min (bb1[0]+bb1[2],bb2[0]+bb2[2])
    y_bottom  = min (bb1[1]+bb1[3],bb2[1]+bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    w = bb1[2]
    h = bb1[3]
    wgt =bb2[2]
    hgt = bb2[3]
    bb1_area = w*h
    bb2_area = wgt*hgt
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou

def drawrandomimagewithbbox(results0,results1,results2,results3,annotations,filenames,randomnumber):
    detectionsres0 = results0.get("bbox")[randomnumber]
    detectionsres1 = results1.get("bbox")[randomnumber]
    detectionsrestemp = results2.iloc[randomnumber]
    detectionsres2 = detectionsrestemp[1:5]
    detectionsrestemp1 = results3.iloc[randomnumber]
    detectionsres3 = detectionsrestemp[1:5]
    
    imagegt = annotations[randomnumber]['bbox']
# #     D = eudist(results,annotations,randomnumber)
    img0=load_image(filenames[randomnumber], imagespath)
    draw_bounding_box(img0,imagegt,detectionsres0,detectionsres1,detectionsres2,detectionsres3)
#     title = "Euclidean distance gt & detection is = "+ str(D)
    title = "Detection Results"
    plot_image2(img0,title)

def read_n_to_last_line(filename, n = 1):
    """Returns the nth before last line of a file (n=1 gives last line)"""
    num_newlines = 0
    with open(filename, 'rb') as f:
        try:
            f.seek(-2, os.SEEK_END)    
            while num_newlines < n:
                f.seek(-2, os.SEEK_CUR)
                if f.read(1) == b'\n':
                    num_newlines += 1
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
    return last_line
def readlabelsgt(path):
    output0 = pd.DataFrame()
    arr = os.listdir(path)
    headers = (['cls', 'x', 'y', 'w','h'])
    for i in arr:
        vari = pd.read_csv(path+i,sep=" ",names=headers)
        output0=output0.append(pd.DataFrame(vari.values[0]).T , ignore_index=True)
    output0.columns=headers
    return output0

def readlabels(path):
    output = pd.DataFrame()
    arr = os.listdir(path)
    headers = (['cls', 'x', 'y', 'w','h','conf'])
    for i in arr:
        vari = pd.read_csv(path+i,sep=" ",names=headers)
        output=output.append(pd.DataFrame(vari.values[0]).T , ignore_index=True)
        if i == arr[-1]:
            speed=read_n_to_last_line(path+i, n = 1)
    output.columns=headers
    return output,speed
def getioutxt(bboxres,bboxgt):
    ioures_v=[]
    for i in range(0,len(bboxgt)):
        ioures_v.append(get_iou2(bboxres[i], bboxgt[i]))
    return ioures_v
def readlabelsbothgt(pathres,pathgt):
    output0 = pd.DataFrame()
    output1 = pd.DataFrame()
    arr1 = os.listdir(pathres)
    arr2 = os.listdir(pathgt)
    arr = [x for x in arr1 if x in arr2]
    headers =  (['cls', 'x', 'y', 'w','h','conf'])
    headersgt = (['cls', 'x', 'y', 'w','h'])
    for i in arr:
                vari = pd.read_csv(pathres+i,sep=" ",names=headers)
                output0=output0.append(pd.DataFrame(vari.values[0]).T , ignore_index=True)
                vari1 = pd.read_csv(pathgt+i,sep=" ",names=headersgt)
                output1=output1.append(pd.DataFrame(vari1.values[0]).T , ignore_index=True)
    output0.columns=headers
    output1.columns=headersgt
    return output0,output1
                   
                   


