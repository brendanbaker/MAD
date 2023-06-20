import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine as scipy_cosine
import numpy as np
from datetime import datetime
import os

def MAD(data_path, space_folder, write = True):
    '''
    Completes all MAD calculations for all spaces.
    '''

    # Load stop words
    stop_words = set(stopwords.words('english'))
    
    def preprocess_text(text):
        '''
        Optimized preprocessing for MAD calculations.
        '''
        text = text.lower()
        text = re.sub(r'\W|\s+[a-zA-Z]\s+|\^[a-zA-Z]\s+|\s+', ' ', text)
        word_tokens = word_tokenize(text)

        # Combine stopwords removal and single character removal
        text = ' '.join(word for word in word_tokens if word not in stop_words and len(word) > 1)

        return text
    
    # List all files in the space folder that end in .csv
    spaces = [file for file in os.listdir(space_folder) if file.endswith(".csv")]
    
    results = []
    data = pd.read_csv(data_path)
    
    for space in spaces:
        space_df = pd.read_csv(space_folder + space)

        # Preprocessing steps
        data['unique'] = range(1, data.shape[0] + 1)
        original = data.copy()
        data = data[['id', 'item', 'response', 'unique']]
        item = data['item'].drop_duplicates().to_frame()
        item = item.merge(space_df, left_on='item', right_on='word', how='left').drop(columns=['word']).rename(columns={'vector': 'item_vector'})
        data['response'] = data['response'].apply(lambda x: preprocess_text(x))
        data['response'] = data['response'].apply(lambda x: word_tokenize(x))
        data = data.explode('response')
        data = data.merge(space_df, left_on='response', right_on='word', how='left').drop(columns=['word']).rename(columns={'vector': 'response_vector'})
        

        # Distance calculation
        distance_list = []
        for i in range(len(data)):
            item_name = data['item'][i]
            item_vector = np.array(item[item['item'] == item_name].iloc[0, 1:]).astype(float)
            response_vector = np.array(data.iloc[i, 4:]).astype(float)
            
            if np.isnan(response_vector).any():  # If there are any NaN in the response vector
                distance = np.nan
            else:
                try:
                    distance = scipy_cosine(item_vector, response_vector)
                except ValueError:
                    distance = np.nan
            distance_list.append(distance)

        # Final steps
        data['distance'] = distance_list
        data = data.groupby('unique').agg({'distance': 'max'})
        data = data[['distance']].merge(original, on='unique', how='left').rename(columns={'distance': 'MAD_' + space[:-4]})
        trimmed = data[['unique', 'id', 'item', 'response', 'MAD_'+ space[:-4]]]
        if write and space[:-4]:
            data.to_csv("MAD_" + space[:-4] + ".csv", index=False)
        results.append(trimmed)

    # Merge all data frames in the list together
    for i in range(len(results)):
        if i == 0:
            final = results[0]
        else:
            final = final.merge(results[i], on=['id', 'item', 'response', 'unique'], how='outer')

    original = pd.read_csv(data_path)
    all_data = pd.merge(final, original, on=['id', 'item', 'response'], how='outer')
    
    if write: all_data.to_csv("MAD_" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + ".csv", index=False)
    
    return all_data
    
        
if __name__ == '__main__':
    results = MAD(data_path = "./data/test_data.csv", space_folder = "./spaces/")
    print(results)