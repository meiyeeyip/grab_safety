# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:25:48 2019

@author: meiyip
"""

import grab_code as models
import pandas as pd
import pickle


filename = 'C:/Users/meiyip/Desktop/grab_safety/model.sav' # please put the file path for model_sav here
model = pickle.load(open(filename, 'rb'))
test_data_path = 'C:/Users/meiyip/Desktop/safety/features/part-00000-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv' # Plese put the test data path here'


test_data = pd.read_csv(test_data_path)


def predict(test_data, model):
        data = models.remove_column(test_data, 'bookingID')
        data = models.create_feature(data)
        result = list(model.predict(data))
        return result
        
result = predict(test_data, model)

