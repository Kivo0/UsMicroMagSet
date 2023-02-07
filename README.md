## All information and downloadable files presented here are only for review purposes. Any Usage for this dataset is not permitted. 
***

# USMicroMagSet: Using deep learning analysis to benchmark the performance of microrobots in ultrasound images.

### We are presenting a US image microrobots dataset, that consists of 8 microrobots grouped in 3 Locomotion principals. 

* Microrobot Class: SMF - Steering magnetic field
    1. Sphere
    2. Cube 
    3. Cylinder
* Microrobot Class: RMF - Rotating magnetic field
    1. Helical 
    2. Soft Sheet
    3. Rolling Cube
* Microrobot Class: OMF - Oscilating magnetic field. 
    1. Flagella
    2. Chainlike Robot - (Sphere3 folder)

We have recorded the microrobots back and forth in a microfludic channel using 14L5 SP Ultrasound probe.
The videos are 10 min each for each microrobot.

Contribution to the dataset guidelines will be defined soon. 



### The dataset consists of 40K images, this github has 2 branches. Branch 2 is for the Tracking format of the same dataset, to enable the community to either build their own image detector or image tracker.   

### we have created a module to visualize the dataset called USMMgSt included with the dataset.

    

## Dataset Samples:


```python
labels , images = USMMgSt.readrobot("cube",3)
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(40,20))
for a in ax.ravel():
    a.axis('off')

for i in range(1,7):
    plt.subplot(2, 3, i)
   
    USMMgSt.plotimage_withbbox(images[i],labels.iloc[i][1:5])
```


    
![png](README_files/README_3_0.png)
    



```python
labels , images = USMMgSt.readrobot("helical",1)
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(40,20))
for a in ax.ravel():
    a.axis('off')

for i in range(1,7):
    plt.subplot(2, 3, i)
   
    USMMgSt.plotimage_withbbox(images[i],labels.iloc[i][1:5])
```


    
![png](README_files/README_4_0.png)
    



```python
labels , images = USMMgSt.readrobot("cylinder",3)
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(40,20))
for a in ax.ravel():
    a.axis('off')

for i in range(1,7):
    plt.subplot(2, 3, i)
   
    USMMgSt.plotimage_withbbox(images[i],labels.iloc[i][1:5])




```


    
![png](README_files/README_5_0.png)
    



```python
labels , images = USMMgSt.readrobot("rollingcube",3)
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(40,20))
for a in ax.ravel():
    a.axis('off')

for i in range(1,7):
    plt.subplot(2, 3, i)
   
    USMMgSt.plotimage_withbbox(images[i],labels.iloc[i][1:5])
```


    
![png](README_files/README_6_0.png)
    



```python
labels , images = USMMgSt.readrobot("flagella",3)
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(40,20))
for a in ax.ravel():
    a.axis('off')

for i in range(1,7):
    plt.subplot(2, 3, i)
   
    USMMgSt.plotimage_withbbox(images[i],labels.iloc[i][1:5])
```


    
![png](README_files/README_7_0.png)
    



```python
labels , images = USMMgSt.readrobot("sheetrobot",3)
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(40,20))
for a in ax.ravel():
    a.axis('off')

for i in range(1,7):
    plt.subplot(2, 3, i)
   
    USMMgSt.plotimage_withbbox(images[i],labels.iloc[i][1:5])
```


    
![png](README_files/README_8_0.png)
    



```python
import USMMgSt
import importlib
importlib.reload(USMMgSt)
```




    <module 'USMMgSt' from 'g:\\deeplearning\\dataset_tracking\\DatasetBackup\\USMMgSt.py'>




```python
labels , images = USMMgSt.readrobot("sphere3",3)
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(40,20))
for a in ax.ravel():
    a.axis('off')

for i in range(1,7):
    plt.subplot(2, 3, i)
   
    USMMgSt.plotimage_withbbox(images[i],labels.iloc[i][1:5])


```


    
![png](README_files/README_10_0.png)
    



```python
# labels , images = USMMgSt.readrobot("sphere1",3)
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(40,20))
for a in ax.ravel():
    a.axis('off')

for i in range(1,7):
    plt.subplot(2, 3, i)
   
    USMMgSt.plotimage_withbbox(images[i+2],labels.iloc[i+2][1:5])
```


    
![png](README_files/README_11_0.png)
    


```python

```
