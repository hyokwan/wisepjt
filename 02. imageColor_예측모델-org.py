#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import os
# load, split and scale the maps dataset ready for training
from os import listdir
warnings.filterwarnings(action="ignore")
# example of loading a pix2pix model and using it for one-off image translation
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,array_to_img
from tensorflow.keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
import matplotlib.pyplot as plt


# In[2]:


import tensorflow as tf


# In[3]:


LOAD_IMG_SIZE = 512


# In[4]:


### 정의: 이미지를 불러온다. 사이즈 256*256
### 파라미터: filename: 예측대상 이미지 파일명
###          size: 불러옫 대상 이미지 사이즈
def load_image(filename, size=(LOAD_IMG_SIZE,LOAD_IMG_SIZE)):
    # load image with the preferred size
    pixels = load_img(filename, target_size=size)
    # convert to numpy array
    pixels = img_to_array(pixels)
    # scale from [0,255] to [-1,1]
    pixels = (pixels - 127.5) / 127.5
    # reshape to 1 sample
    pixels = expand_dims(pixels, 0)
    
    return pixels


# ### 1.4 저장된 모델 활용 예측

# ### 1.4.1 이미지 불러오기

# In[5]:


# # 테스트 이미지 불러오기
# src_image = load_image("./문화유산 이미지 자료/흑백원본_문화재청제공/187266_00_232_35.jpg")
# print('Loaded', src_image.shape)


# In[6]:


testImgFullPath = "./문화유산 이미지 자료/test3/1095.jpg"


# In[7]:


# 테스트 이미지 불러오기
src_image = load_image(testImgFullPath)
print('Loaded', src_image.shape)


# ### 1.4.2 모델 불러오기

# In[8]:


##### 변경 포인트 ##########################
# 저장된 훈련 모델 불러오기
model = load_model('model_gogung_day.h5')


# In[9]:


# 테스트 이미지 불러오기
srcImage = load_img(testImgFullPath)
print('Loaded', src_image.shape)


# In[10]:


# 예측하기
gen_image = model.predict(src_image)


# In[11]:


plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.imshow(srcImage)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(gen_image[0])
plt.axis('off')


# ### 최종 테스트 결과 폴더 데이터 예측

# In[12]:


testFolder = "./문화유산 이미지 자료/test3/"

testImgs = listdir(testFolder)


# In[13]:


targetFolder = "./문화유산 이미지 자료/result3/"


# In[14]:


import os
if not os.path.exists(targetFolder):
    os.makedirs(targetFolder)


# In[15]:


testImgs[3]


# In[ ]:


for i in range(0, len(testImgs)):
    try:
        testImgPath = os.path.join(testFolder,testImgs[i])
        srcImage =             load_img(testImgPath)

        # 테스트 이미지의 가로/세로 해상도 가져오기
        orgWidth = img_to_array(srcImage).shape[1]
        orgHeight = img_to_array(srcImage).shape[0]

        # 예측을 위한 이미지 로딩 및 전처리
        srcImageClean = load_image(testImgPath)

        # 예측 후 이미지 픽셀값 0~1로 변환
        gen_image = model.predict(srcImageClean)
        gen_image = (gen_image + 1) / 2.0 #convert 0~1value
        # nrows=2, ncols=1, index=1
        # 예측 이미지 원본 이미지 사이즈로 변환
        gen_image = gen_image.reshape(LOAD_IMG_SIZE,LOAD_IMG_SIZE,3)
        new_image = tf.image.resize(gen_image, (orgHeight, orgWidth))

        # 예측결과 저장
        cvtedImgObj = array_to_img(new_image)
        targetFullPath = os.path.join( targetFolder, testImgs[i])
        cvtedImgObj.save(targetFullPath)

        # 시각화
        plt.figure(figsize=(15,5))
        plt.subplot(1, 2, 1)
        plt.imshow(srcImage)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)
        plt.axis('off')
    except Exception as e:
        pass
        print(e)

