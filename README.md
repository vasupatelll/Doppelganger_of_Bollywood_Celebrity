## 📌 Project Description:

This project is aimed at developing a deep learning system that can detect and recognize people who look identical to Bollywood actors. 
It will use facial features as the basis of comparison for detecting similar features between two people. 
This system could be used in various applications such as security systems, entertainment platforms, etc., 
where it would help identify individuals with similar facial features to famous personalities. 

The development process of this project involves several steps including image cleaning and preparation, model training and testing, 
deployment on the streamlit, and integration into existing systems or applications 

## ⚙️Functionality :
- In this project, I employ the state-of-the-art VGGface model, which is a powerful deep convolutional neural network specifically designed for face recognition tasks. By utilizing this model, I can extract high-level facial features from the image dataset. The VGGface model is known for its ability to capture intricate details and unique facial characteristics with precision.

- Once the features are extracted using the VGGface model, I save them into a pickle file. This file format allows for efficient storage and retrieval of the features without having to recompute them every time. By saving the features in a pickle file, I can reduce computation time and ensure quick access to the necessary information.

## <b>Recommendation System :</b>
To recommend the top 5 nearest images similar to the input image, I employ the NearestNeighbour algorithm along with cosine_similarity in my model. Here's how the recommendation system works:

- I compute the cosine similarity between the feature vectors extracted from the image dataset and the input image. Cosine similarity measures the angle between two vectors and provides a score that represents their similarity. A higher score indicates a greater similarity between the feature vectors.

- By utilizing the NearestNeighbour algorithm, I identify the images that have the highest cosine similarity scores. This algorithm helps me find the nearest neighbors based on the calculated cosine similarity.

- Using the sorted list of images based on the cosine similarity scores, I present the top 5 nearest images to the user as recommendations. These images are deemed the most similar to the input image based on their feature vectors, and are thus likely to share similar characteristics and visual patterns.

## <b>Application in Multi-National Companies :</b>
- <b>Content Curation for Marketing:</b> Companies can utilize the recommendation system to curate image content for marketing purposes. By inputting an image related to their brand or product, the system can recommend visually similar images that can be used for marketing campaigns, brand promotion, and advertising.

- <b>Image Search and Tagging:</b> With a large image dataset, multi-national companies often struggle with efficient image search and categorization. The project's recommendation system can aid in searching for visually similar images, making it easier to find specific images and tag them accordingly. This streamlines the image management process.

- <b>Product Recommendation:</b> In e-commerce companies, the project's recommendation system can be used to recommend visually similar products to customers. By inputting an image of a product, the system can suggest other visually similar products, thereby increasing cross-selling and upselling opportunities.

## 📜 LICENSE
[MIT](LICENSE)
