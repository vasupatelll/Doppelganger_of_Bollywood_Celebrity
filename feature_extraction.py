################## IMPORTS #####################
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm
import os

################## EXTRACTING FILE NAMES #####################

# I am extracting file name because every bollywoood actors in the data had their specific folder
actors = os.listdir('Data_of_bollywood_celeb_face_localized')

filenames = []

for actor in actors:
    for file in os.listdir(os.path.join('Data_of_bollywood_celeb_face_localized', actor)):
        filenames.append(os.path.join('Data_of_bollywood_celeb_face_localized', actor, file))

# Dumping into pickle file so that aferwards I can excess their features easily
pickle.dump(filenames,open('filename.pkl', 'wb'))

# Opening pickle file which I had saved earlier
filenames = pickle.load(open('filename.pkl', 'rb'))


################## BUILDING MODEL #####################

model = VGGFace(model='resnet50',
                include_top=False,
                input_shape=(224, 224, 3),
                pooling='avg')

################## CREATING FEATURE EXTRACTION FUNCTION #####################
# Creating feature extraction function to extract features form image

def feature_extractor(img_path, model):
    img = image.load_img(img_path,target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()

    return result

features = []

for file in tqdm(filenames):
    features.append(feature_extractor(file, model))

# Dumping inmages features into pickle file
pickle.dump(features, open('Embedding.pkl', 'wb'))