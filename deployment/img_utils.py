import streamlit as st
import skimage.color as sic
import tensorflow as tf

from tensorflow.keras.utils import img_to_array
import numpy as np

def prepare_img(img):
  if len(img.shape)==3:
    if img.shape[-1]==4:
      img=img[:,:,0:3]
    img=sic.rgb2gray(img)*255
  img=img_to_array(img)
  return img
@st.cache_resource
def load_as_tensor(img):
  img=prepare_img(img)
  img=np.repeat(img, 3, axis=-1)    
  img=img/255.0
  img=tf.image.resize(img, [256, 256])
  img=np.expand_dims(img,axis=0)
  tensor=tf.convert_to_tensor(img)
  return tensor