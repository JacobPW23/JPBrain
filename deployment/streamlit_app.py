import streamlit as st
import os
from keras.models import load_model
from PIL import Image
from io import BytesIO
import numpy as np
from  img_utils import *

MODEL_PATH='../models'
TUMOR_LABELS=np.load('tumor_labels.npy')

def get_models():
    models=[]
    for dirname,_,model_names in os.walk(MODEL_PATH):
        for name in model_names:
            models.append(name)
    return models

st.set_page_config(layout="wide")
left,right=st.columns(2)

@st.cache_resource
def get_model(name):
    return load_model(MODEL_PATH+'/'+name)

controls=left.container()
results=right.container()
with  controls:
    st.title("JPBrain")
    st.markdown("Aplikacja do klasyfikacji zdjęć MRI/CT pod względem rodzaju występujących nowotworów")
    selected=st.selectbox('Wybierz model:',get_models())  
    uploaded=st.file_uploader('Wybierz plik z obrazem mózgu:')
    model=None
    try:
        model=get_model(selected)
    except ValueError:
        st.warning('Model nieobsługiwany')
    if uploaded and model:
        bytes_data = uploaded.getvalue()
        img = Image.open(BytesIO(bytes_data))
        img=np.asarray(img)
        button=st.button('Diagnozuj')
        if button :
            label_indexes=model.predict(load_as_tensor(img))
            st.markdown('#### Diagnoza: {:s}'.format(TUMOR_LABELS[np.argmax(label_indexes)])
                        +'\nPewność: {:.000%}'.format(np.floor(np.max(label_indexes)*100)/100.0))

        with results:
            if uploaded is not None and img is not None:
                st.image(img,use_container_width=True)
        
       
        
        
