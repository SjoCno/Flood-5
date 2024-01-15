# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:23:10 2023

@author: derby
"""

import pickle


def load_dataset(dataset_name, dataset_folder='datasets/'):
    '''
    Loads dataset, composed by a list of pytorch geometric data objects
    only accepts files of .pkl format
    ------
    dataset_name: str
        name of the dataset to be loaded
    '''
      
    path = f"{dataset_folder}/{dataset_name}.pkl"
    
    with open(path, 'rb') as file:
        dataset = pickle.load(file)
    
    return dataset