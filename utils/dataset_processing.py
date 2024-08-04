import os
import numpy

import streamlit as st

from sklearn import preprocessing
from sklearn.svm import SVC

from utils.face_processing import load_faces_from_train_val_prod, load_faces_prod, load_faces_with_path

def loading_data():
    path = "./images/mv"
    # change the working directory to the path where the images are located

    # this list holds all the image filename
    flowers = []

    # creates a ScandirIterator aliased as files
    with os.scandir(path) as files:
    # loops through each file in the directory
        for file in files:
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
            if file.name.endswith(image_extensions):
            # adds only the image files to the flowers list
                flowers.append(os.path.join(path, file.name))

    return flowers


def save_dataset(datasize_name, train, val):
    with st.spinner('saving ...'):
        trainX, trainy = load_faces_from_train_val_prod(train)
        testX,  testy  = load_faces_from_train_val_prod(val)
        # metadata = {
        #     'train_dir': train,
        #     'val_dir': val,
        #     'num_train_samples': len(trainX),
        #     'num_val_samples': len(testX)
        # }
        # save arrays to one file in compressed format
        numpy.savez_compressed(datasize_name, trainX, trainy, testX, testy)
    return datasize_name


def load_dataset_with_metadata(dataset_name):
    with numpy.load(dataset_name, allow_pickle=True) as data:
        trainX = data['trainX']
        trainy = data['trainy']
        testX = data['testX']
        testy = data['testy']
        # metadata = data['metadata'].item()  # .item() to convert array to dictionary
    return trainX, trainy, testX, testy


def save_dataset_prod(dataset, face_folder):
    # save one set of dataset
    
    with st.spinner('saving ...'):
        trainX, trainy, file_names = load_faces_with_path(face_folder)
        st.write(file_names)
        # metadata = {
        #     'train_dir': train,
        #     'val_dir': val,
        #     'num_train_samples': len(trainX),
        #     'num_val_samples': len(testX)
        # }
        # save arrays to one file in compressed format
        numpy.savez_compressed(dataset, trainX, trainy, file_names)
    return dataset

