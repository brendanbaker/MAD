import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine as scipy_cosine
import numpy as np
from datetime import datetime
import os

def preprocess_text(text):
    '''
    Preprocesses text for MAD calculations.
    
    '''
    # Lowercase and punctuation
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    # Stop words
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    text = ' '.join([word for word in word_tokens if word.casefold() not in stop_words])
    
    # Remove single characters
    text = ' '.join([word for word in text.split() if len(word) > 1])
    
    return text


def MAD(data_path, space_path, label = "space", write = True):
    '''
    Calculates Maximal Associative Distance (MAD) scores for each response.
    '''
    data = pd.read_csv(data_path)
    space = pd.read_csv(space_path)
    

    # Add a unique identifier for each response
    data['unique'] = range(1, len(data) + 1)

    # Save original 
    original = data.copy()
    
    # Keep only id, item, response and unique columns
    data = data[['id', 'item', 'response', 'unique']]
    
    # Find the item vector: separate the item column from the data then append the space that matches the item in the word column 
    item = data['item'].drop_duplicates()
    item = item.to_frame()
    item = item.merge(space, left_on='item', right_on='word', how='left')
    item = item.drop(columns=['word'])
    item = item.rename(columns={'vector': 'item_vector'})
    
    # Run text preprocessing on the responses
    data['response'] = data['response'].apply(lambda x: preprocess_text(x))
    
    # Now tokenize the response column, pivot the data, and merge the space that matches the word in the response column
    data['response'] = data['response'].apply(lambda x: word_tokenize(x))
    data = data.explode('response')
    data = data.merge(space, left_on='response', right_on='word', how='inner')
    data = data.drop(columns=['word'])
    data = data.rename(columns={'vector': 'response_vector'})
    
    # Loop through every row in the dataframe
    distance_list = []

    for i in range(len(data)):
        # Get the item name
        item_name = data['item'][i]
        
        # Find the item vector
        item_vector = item[item['item'] == item_name]
        
        # Take all values in that row that are in a "V" column, store as np array
        item_vector = np.array(item_vector.iloc[0, 1:])
        
        # Get the response vector - these are all the columns after id, item, response, and unique
        response_vector = np.array(data.iloc[i, 4:])
        
        # Make sure we are dealing with numeric values only
        try:
            item_vector = item_vector.astype(float)
        except ValueError:
            print("Non-numeric data found in item_vector")

        try:
            response_vector = response_vector.astype(float)
        except ValueError:
            print("Non-numeric data found in response_vector")
        
        # Get the cosine distance between the item vector and the response vector
        distance = scipy_cosine(item_vector, response_vector)
        
        # Add the distance to the dataframe
        distance_list.append(distance)
    
    # Append the distance list 
    data['distance'] = distance_list

    # Select the maximum distance from each unique identifier
    #data = data.groupby('unique').max()
    
    # Do it another way to avoid deprecation warning
    data = data.groupby('unique').agg({'distance': 'max'})
    
    # Select distance and unique, then merge with original dataframe
    data = data[['distance']]
    data = data.merge(original, on='unique', how='left')
    
    # Rename distance to MAD_label
    data = data.rename(columns={'distance': 'MAD_' + label})
    
    # Select unique, id, item, response, and distance, return this if write = False
    trimmed = data[['unique', 'id', 'item', 'response', 'MAD_'+label]]
    
    # Write the data to a csv
    if write:
        if label:
            data.to_csv("MAD_" + label + ".csv", index=False)
        
        else:
            data.to_csv("MAD_" + str(datetime.now()) + ".csv", index=False)
        return data
    else:
        return trimmed
        
    

def all_mad(data_path, space_folder, write = True):
    '''
    Completes all MAD calculations for all spaces.
    '''
    
    # Get the original columns
    original = pd.read_csv(data_path)
    
    # List all files in the space folder that end in .csv
    spaces = [file for file in os.listdir(space_folder) if file.endswith(".csv")]
    
    # Iterate through each space and run MAD
    results = [MAD(data_path=data_path, space_path=space_folder + space, label=space[:-4], write = False) for space in spaces]
    
    # Merge all data frames in the list together
    for i in range(len(results)):
        if i == 0:
            final = results[0]
        else:
            final = final.merge(results[i], on=['id', 'item', 'response', 'unique'], how='outer')
            
    
    # Add original columns
    all_data = pd.merge(final, original, on=['id', 'item', 'response'], how='outer')
    
    # Write to a csv if desired
    if write: all_data.to_csv("MAD_" + str(datetime.now()) + ".csv", index=False)
    
    # Return the results
    return all_data
    
        
if __name__ == '__main__':
    results = all_mad(data_path = "./data/test_data.csv", space_folder = "./spaces/")
    print(results)
