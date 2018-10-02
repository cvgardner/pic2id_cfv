import requests
from PIL import Image
from io import BytesIO
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
import numpy as np
import pandas as pd
from tqdm import tqdm 
import nmslib


def FeaturesFromUrl(model, url):
    '''Takes an input URL which contains only an image and returns a feature vector extracted by using the input keras model

    model: keras model for feature extraction
    url: url string of image location'''
    #pull image from url
    page = requests.get(url)
    img = Image.open(BytesIO(page.content)).convert("RGB")

    #pull out feature vector from image
    img = img.resize((224,224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    features = model.predict(arr)
    return features

def GetFeatureList(model):
    '''creates a featurelist and corresponding card id

    model: keras model for feature extraction

    returns
    flist: list of lists containing feature vectors from the model
    idlist: list of card ids with the same indexing as flist'''
    
    #Will have to use multiple loops to deal with changes in url formatting
    urlbase = "https://en.cf-vanguard.com/jsp-material/cardimages/"
    flist = []
    idlist = []
    
    #loops for original series
    for i in tqdm(range(1,18)): #original booster sets
        for j in range(200): #overcompensate for amount of cards in a set
            try:
                url=urlbase+"bt_{:02d}_{:03d}.jpg".format(i,j)
                features = FeaturesFromUrl(model,url)
                flist.append(features[0])
                idlist.append("bt{:02d}_{:03d}".format(i,j))
            except:
                pass

    #loops for gbt-jpg
    for i in tqdm(range(1,4)): #original booster sets
        for j in range(200): #overcompensate for amount of cards in a set
            try:
                url=urlbase+"gbt{:02d}_{:03d}.jpg".format(i,j)
                features = FeaturesFromUrl(model,url)
                flist.append(features[0])
                idlist.append("gbt{:02d}_{:03d}".format(i,j))
            except:
                pass
   
    #loops for gbt-png
    for i in tqdm(range(5,15)): #original booster sets
        for j in range(200): #overcompensate for amount of cards in a set
            try:
                url=urlbase+"gbt{:02d}_{:03d}.png".format(i,j)
                features = FeaturesFromUrl(model,url)
                flist.append(features[0])
                idlist.append("gbt{:02d}_{:03d}".format(i,j))
            except:
                pass
   
    #loops for vbt
    for i in tqdm(range(1,2)): #original booster sets
        for j in range(200): #overcompensate for amount of cards in a set
            try:
                url=urlbase+"vbt{:02d}_{:03d}.png".format(i,j)
                features = FeaturesFromUrl(model,url)
                flist.append(features[0])
                idlist.append("vbt{:02d}_{:03d}".format(i,j))
            except:
                pass

    return flist, idlist

def BuildSaveIndex(flist):
    '''builds and saves an index using a featurelist'''
    index = nmslib.init(method='hnsw', space = 'l2')
    index.addDataPointBatch(flist)
    index.createIndex({'post':2})
    index.saveIndex('cfv_index.bin')



