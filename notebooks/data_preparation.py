import os
import numpy as np
import pandas as pd

root_directory = '/data/CheXpert-v1.0'
valid_folder = os.path.join(root_directory, 'valid')

chexnet_targets = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                   'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                   'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                   'Support Devices']

chexpert_targets = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

u_one_features = []
u_zero_features = chexnet_targets 

def label_u_one_features(df: pd.DataFrame, column: str):
    return np.where((df[column] == 1) | (df[column] == -1), 1, 0)

def label_u_zero_features(df: pd.DataFrame, column: str):
    return np.where(df[column] == 1, 1, 0)

def feature_string(row):
    feature_list = []
    for feature in u_one_features:
        if row[feature] in [-1,1]:
            feature_list.append(feature)
            
    for feature in u_zero_features:
        if row[feature] == 1:
            feature_list.append(feature)
            
    return ';'.join(feature_list)

def prepare_data():
    valid_labels_df = pd.read_csv(os.path.join(root_directory, 'valid.csv'))

    valid_labels_df['patient'] = valid_labels_df['Path'].apply(lambda x: os.path.split(os.path.split(os.path.split(x)[0])[0])[1])
    valid_labels_df['study'] = valid_labels_df['Path'].apply(lambda x: os.path.split(os.path.split(x)[0])[1])

    valid_labels_df['feature_string'] = valid_labels_df.apply(feature_string,axis = 1).fillna('')
    valid_labels_df['feature_string'] = valid_labels_df['feature_string'].apply(lambda x:x.split(";"))

    for col in u_one_features:
        valid_labels_df[f'{col}_u_one'] = label_u_one_features(df=valid_labels_df, column=col)
    for col in u_zero_features:
        valid_labels_df[f'{col}_u_zero'] = label_u_zero_features(df=valid_labels_df, column=col)

    valid_labels_df['path'] = '/data/' + valid_labels_df['Path']
    targets = [target + '_u_zero' for target in u_zero_features]

    X_val = valid_labels_df['path'].values
    y_val = valid_labels_df[targets].values

    return X_val, y_val, valid_labels_df, targets