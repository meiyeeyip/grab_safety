# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:44:29 2019

@author: meiyip
"""

import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import math
import pickle

path = r'C:/Users/meiyip/Desktop/safety/features'
label_path = 'C:/Users/meiyip/Desktop/safety/labels/part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv'



def import_data(path, label_path):
        all_files = glob.glob(path + "/*.csv")
        # import labels data
        label = pd.read_csv(label_path)
        return all_files, label


def data_cleaning(all_files, label):
        # combine all the csv file
        filename_list = []
        
        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            filename_list.append(df)
        
        # concate data
        data = pd.concat(filename_list, axis=0, ignore_index=True)
        
        # left join features and label
        data_merge = pd.merge(data, label, on='bookingID', how='left')

        return data_merge
    
def remove_column(data, column_name):
        del data[column_name]
        return data


def create_feature(data):
        # feature 1 (GPS related)
        data['bearing_accuracy'] = data['Accuracy']*data['Bearing']
        data['bearing_accuracy_speed'] = data['Accuracy']*data['Bearing']*data['Speed']
        
        # create feature general acceleration
        data['acceleration'] = data['acceleration_x']**2 + data['acceleration_y']**2 + data['acceleration_z']**2
        data['acceleration']= (data['acceleration']).astype(float)
        data['acceleration'] = data['acceleration'].apply(math.sqrt)
        
        # create feature distance
        data['distance'] = data['Speed']*data['second']
        
        # create  feature velocity
        data['velocity'] = data['acceleration']*data['second']
        return data



def data_split(data):
        # Split features and label
        X = data.loc[:, data.columns != 'label']
        y = data.loc[:, data.columns == 'label']
        
        # split training and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        
        # split out validation set
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
        
        return X_train, X_val, X_test, y_train, y_val, y_test



def model_training(X_train, X_val, X_test, y_train, y_val, y_test): 

        clf = DecisionTreeClassifier(criterion = "gini",
                    random_state = 150,max_depth=10, min_samples_leaf=15)

        clf.fit(X_train, y_train)
        
        # Data Scoring
        print(clf.score(X_train, y_train))
        print(clf.score(X_val, y_val))
        print(clf.score(X_test, y_test))
        
        return clf


def create_model(path, label_path):
        all_files, label = import_data(path, label_path)
        data_merge = data_cleaning(all_files, label)
        data_merge = remove_column(data_merge, 'bookingID')
        data_new_feature = create_feature(data_merge)
        X_train, X_val, X_test, y_train, y_val, y_test = data_split(data_new_feature)
        model = model_training(X_train, X_val, X_test, y_train, y_val, y_test)
        return model


if __name__=='__main__':
        model = create_model(path, label_path)
        # save model
        filename = 'C:/Users/meiyip/Desktop/grab_safety/model.sav'
        pickle.dump(model, open(filename, 'wb'))

