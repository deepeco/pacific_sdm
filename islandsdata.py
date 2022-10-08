
import numpy as np
import pandas as pd
import gc
import os
from IPython.core.display import display, HTML
from bin.model import *
from sklearn.pipeline import Pipeline
from bin.conf import PREDICTOR_LOADERS
from bin.loader import get_predictor_data
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.manifold import MDS
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
from collections import defaultdict
import matplotlib
from skimage.measure import label

os.environ['PROJ_LIB'] = '/home/dmitry/bin/gdal/share/proj'

BBOX = [43.028343, 145.372467, 51.042510, 157.320565]
NRES = 3000

latitude, longitude = np.meshgrid(np.linspace(BBOX[0], BBOX[2], NRES),      
                                  np.linspace(BBOX[1], BBOX[-1], NRES))

VARIABLES = [f'BIO{k}' for k in range(1, 20)] + ['WKI5', 'PCKI0', 'PWKI0', 'CKI5', 'IC']

array = get_predictor_data(tuple(latitude.ravel()), tuple(longitude.ravel()),
                            'BIO1', postfix='')


array = array.reshape(NRES, NRES)
array = ~np.isnan(array)
array_labels = label(array, connectivity=2)

print(f"Total number of connected regions: {len(np.unique(array_labels.ravel()))}.")

def get_data(vars):
    for l in np.unique(array_labels.ravel()):
        data = {'latitude': latitude[array_labels == l].ravel(),
                'longiture': longitude[array_labels == l].ravel()}
        print("Current region is ", l)
        print("The number of points: ", len(data["latitude"]))
        for j in vars:
            array = get_predictor_data(tuple(latitude[array_labels == l].ravel()), tuple(longitude[array_labels == l].ravel()),
                            j, postfix='')
            print("Prepared for ", j)
            if np.isnan(array[0]):
                print("dropping for label, ",l)
                break
            data.update({j:  array})
        if 'BIO1' not in data:
            continue
        df = pd.DataFrame(data)
        df.drop_duplicates(subset=df.columns[2:]).to_csv(f'data_{l}.csv')

   
        
get_data(VARIABLES)