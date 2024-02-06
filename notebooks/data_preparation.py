import os
import numpy as np
import pandas as pd

root_directory = '/data/CheXpert-v1.0'

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

def prepare_data(split: str = 'valid'):
    admissible_split_values = ['train', 'valid']
    assert split in admissible_split_values, f'A split value of {split} was given. The admissible split values are {admissible_split_values}.'
    
    if split == 'valid':
        file = 'valid.csv'
    else:
        file = 'train.csv'
    df = pd.read_csv(os.path.join(root_directory, file))

    df['patient'] = df['Path'].apply(lambda x: os.path.split(os.path.split(os.path.split(x)[0])[0])[1])
    df['study'] = df['Path'].apply(lambda x: os.path.split(os.path.split(x)[0])[1])

    df['feature_string'] = df.apply(feature_string,axis = 1).fillna('')
    df['feature_string'] = df['feature_string'].apply(lambda x:x.split(";"))

    for col in u_one_features:
        df[f'{col}_u_one'] = label_u_one_features(df=df, column=col)
    for col in u_zero_features:
        df[f'{col}_u_zero'] = label_u_zero_features(df=df, column=col)

    df['path'] = '/data/' + df['Path']
    targets_u_one = [target + '_u_one' for target in u_one_features]
    targets_u_zero = [target + '_u_zero' for target in u_zero_features]
    targets = targets_u_one + targets_u_zero

    X = df['path'].values
    y = df[targets].values

    return X, y, df, targets