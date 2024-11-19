## 📌 Project Description:

This project involves building a deep learning model that can identify and classify Bollywood lookalikes. It will use the facial features to compare and identify whether it is two persons. This new approach could have numerous uses, ranging from security to entertainment with possibilities for identifying people matching a certain face type as that of particular celebrities. Phases of the Development ProcessImage Cleaning and PreparationModel Training & TestingDeployment on StreamlitIntegration into existing systems or applications. 

## ⚙️Functionality :
- I use here VGGface model, which is the most capable deep convolutional neural network for face recognition. Using this model, I am able to forecast the original image data from high-rated facial features. About vggface The VGGFace model is the most accurate precision, it can detect details very fine in face.

- I then save these features into a pickle file after feature extraction from the VGGface model. In this way, the properties can be stored and retrieved fairly fast without needing to calculate them on-the-fly every time. Since I have made Pickle file of features, this will help me to set the things in motion and not wait for long executing time.

## 💡 Recommendation System :
I am using cosine_similarity along with NearestNeighbour algorithm in my model to recommend top 5 nearest images that are similar to the input image. How the recommendation system works:

- I get the cosine similarity using feature vectors extracted from an image dataset and input images. The cosine similarity measures the orientation (that is, frequency of non-zero elements in two vectors) and the magnitude of difference between each element. The more the value its higher is,it signifies that feature vectors are very similar.
  
- Using the NearestNeighbour algorithm, I find out which are those images that have maximum cosine similarity scores. I am then using this algorithm, with which I get the closest neighbors on cosine similarity.
  
- I display the top 5 most similar images as recommendations to users with pictures of certain classes based on their cosine similarity scores in a sorted list. These images are detected as the most similar to the input image in terms of feature vectors and consequently consist common characteristics, probably visual patterns as such.

## 📝 Application in Multi-National Companies :
- <b>Content Curation for Marketing:</b> Brands could benefit by selecting image content for the purpose of marketing from their recommendation system. It allow the user to seed an image associated with their brand or product, and matches it other visual images that can be used in marketing campaigns.

- <b>Image Search and Tagging:</b> With a large image dataset, multi-national companies often struggle with efficient image search and categorization. The project's recommendation system can aid in searching for visually similar images, making it easier to find specific images and tag them accordingly. This streamlines the image management process.

- <b>Product Recommendation:</b> In e-commerce companies, typically have large image datasets, and will often struggle with trying to effectively search for images or categorize the data. A recommendation system of the project helps in finding visually similar images which help you to locate a specific image and categorize better. This helps to simplify the image management process.

## 📜 LICENSE
[MIT](LICENSE)
