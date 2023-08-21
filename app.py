################## IMPORTS #####################
from PIL.Image import Image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np

################## BUILDING MODEL #####################
detector = MTCNN()
model = VGGFace(model='resnet50',
                include_top=False,
                input_shape=(224, 224, 3),
                pooling='avg')


################## LOADING IMAGE FEATURES & FILENAME FEATURES  #####################
feature_list = pickle.load(open('Embedding.pkl', 'rb'))
filenames = pickle.load(open('filename.pkl', 'rb'))


################## SAVING UPLOADED IMAGES INTO SEPERATE FOLDER #####################
def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True

    except:
        return False


################## CREATING FEATURE EXTRACTION FUNCTION #####################
# In this function hyperparameter required are image path, model and detector

def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    #  extract its features
    image = Image.fromarray(face).resize((224, 224))

    face_array = np.asarray(image)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result


################## CREATING RECOMENDATION FUNCTION #####################
def recommend(feature_list, features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    return sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]


################## DEPLOYING PROJECT ON STREAMLIT #####################
st.title('Dopplganger of Bollywood Celebrity')

uploaded_image = st.file_uploader('Upload An Image')

if uploaded_image is not None:
    # save the image in a directory
    if save_uploaded_image(uploaded_image):
        # load the image
        display_image = Image.open(uploaded_image)

        # extract the features
        features = extract_features(os.path.join('uploads', uploaded_image.name), model, detector)
        # recommend
        index_pos = recommend(feature_list, features)
        predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
        # display
        col1, col2 = st.columns(2)

        with col1:
            st.header('Uploaded Image')
            st.image(display_image)
        with col2:
            st.header("Image Appearance Is More Likely To  " + predicted_actor)
            st.image(filenames[index_pos], width=300)