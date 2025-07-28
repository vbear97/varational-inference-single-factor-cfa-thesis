'''Miscallenaous helper functions'''
import pickle
from typing import Any 
def pickle_file(filename: str, my_object: Any): 
    with open(filename, 'wb') as f: 
        pickle.dump(my_object, f)

def unpickle_file(filename: str): 
    with open(filename, 'rb') as f: 
        my_object = pickle.load(f)
        return my_object 