import streamlit as st
from streamlit_extras.stateful_button import button
from utils import streamlit_components, dataset_processing
streamlit_components.streamlit_ui('🦣 Show Training Pictures')
# -------------------------------------------------------------------------------------
import numpy as np
import os
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from random import randint

import tensorflow as tf
import tensorflow_datasets as tfds

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

flowers = dataset_processing.loading_data()     
img = tf.keras.utils.load_img(flowers[0], target_size=(224,224))    #1.1. load the image as a 224x224 array
img = np.array(img)                                                 #1.2. convert from 'PIL.Image.Image' to numpy array
# st.write(img.shape)     # (224, 224, 3)

# Currently, our array has only 3 dimensions (rows, columns, channels) and the model operates in batches of samples. 
# So we need to expand our array to add the dimension that will let the model know how many images we are giving it 
# (num_of_samples, rows, columns, channels).

reshaped_img = img.reshape(1,224,224,3)     
# st.write(reshaped_img.shape)

#1.3. The last step is to pass the reshaped array to the preprocess_input method and our image is ready to be loaded into the model.
x = tf.keras.applications.vgg16.preprocess_input(reshaped_img, data_format=None) 

# Now we can load the VGG model and remove the output layer manually. This means that the new final layer 
# is a fully-connected layer with 4,096 output nodes. This vector of 4,096 numbers is the feature vector that we will use to cluster the images.

model = tf.keras.applications.VGG16()                                           #2.1 load model
model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)    #2.2 remove the output layer

# Now that the final layer is removed, we can pass our image through the predict method to get our feature vector.
features = model.predict(x)
# st.write(features.shape)    #(1,4096)

def extract_features(file, model):
    # load the image as a 224x224 array
    img = tf.keras.utils.load_img(file, target_size=(224,224))    #1.1. load the image as a 224x224 array
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = tf.keras.applications.vgg16.preprocess_input(reshaped_img) 
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features


# Now we can use this feature_extraction function to extract the features from all of the images 
# and store the features in a dictionary with filename as the keys.

data = {}
p = "./data/flower_features.pkl"

# lop through each image in the dataset
for flower in flowers:
    # try to extract the features and update the dictionary
    try:
        feat = extract_features(flower,model)
        data[flower] = feat
    # if something fails, save the extracted features as a pickle file (optional)
    except:
        with open(p,'wb') as file:
            pickle.dump(data,file)
          
 
# get a list of the filenames
filenames = np.array(list(data.keys()))

# get a list of just the features
feat = np.array(list(data.values()))
# st.info(feat.shape)
#  (210, 1, 4096)

# reshape so that there are 210 samples of 4096 vectors
feat = feat.reshape(-1,4096)
# st.info(feat.shape)
# (210, 4096)

# get the unique labels (from the flower_labels.csv)
df = pd.read_csv('./images/flower_images/flower_images/flower_labels.csv')
label = df['label'].tolist()
unique_labels = list(set(label))
# st.text(unique_labels)

# Dimensionality Reduction (PCA)
# Since our feature vector has over 4,000 dimensions, We can't simply just shorten the list by 
# slicing it or using some subset of it because we will lose information. If only there was a way 
# to reduce the dimensionality while keeping as much information as possible.

# Simply put, if you are working with data and have a lot of variables to consider (in our case 4096), 
# PCA allows you to reduce the number of variables while preserving as much information from the original set as possible.

# The number of dimensions to reduce down to is up to you and I'm sure there's a method for finding the best number of 
# components to use, but for this case, I just chose 100 as an arbitrary number.

pca = PCA(n_components=100, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

st.write(f"Components after PCA: {pca.n_components}")

# Now that we have a smaller feature set, we are ready to cluster our images.
# KMeans clustering
# You’ll define a target number k, which refers to the number of centroids you need in the dataset. 
# A centroid is the imaginary or real location representing the center of the cluster.

# This algorithm will allow us to group our feature vectors into k clusters. Each cluster should contain 
# images that are visually similar. In this case, we know there are 10 different species of flowers so we can have k = 10.

kmeans = KMeans(n_clusters=len(unique_labels),random_state=22)
kmeans.fit(x)

st.write(kmeans.labels_)

# Each label in this list is a cluster identifier for each image in our dataset. 
# The order of the labels is parallel to the list of filenames for each image. This way we can group the images into their clusters.

# holds the cluster id and the images { id: [images] }
groups = {}
for file, cluster in zip(filenames,kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)
        
st.write(groups[0])


# function that lets you view a cluster (based on identifier)        
def view_cluster(cluster):
    plt.figure(figsize = (25,25));
    # gets the list of filenames for a cluster
    # files = groups[cluster]
    files = groups.get(cluster, [])
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:29]
    # plot each image in the cluster
    
    # Set up a grid layout
    num_images = len(files)
    cols = 10
    rows = (num_images // cols) + 1 if num_images % cols != 0 else num_images // cols
    
    # Create subplots
    fig, axs = plt.subplots(rows, cols, figsize=(25, 25))
    axs = axs.flatten()
    
    
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1);
        img = tf.keras.utils.load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
        
 
    for index, file in enumerate(files):
        img = tf.keras.utils.load_img(file)
        img = np.array(img)
        axs[index].imshow(img)
        axs[index].axis('off')

    # Hide any remaining subplots that aren't used
    for i in range(index + 1, len(axs)):
        axs[i].axis('off')

    # Display the grid in Streamlit
    st.pyplot(fig)
    
selected_cluster = st.selectbox("Select Cluster", list(groups.keys()))
view_cluster(selected_cluster)