'''
MAD.py
 
Code for Maximum Associative Distance (MAD) algorithm for creativity research
@Brendan Baker

Citation: 
Yu, Y., Beaty, R. E., Forthmann, B., Beeman, M., Cruz, J. H., & Johnson, D. (2023). 
A MAD method to assess idea novelty: Improving validity of automatic scoring using maximum associative distance (MAD). 
Psychology of Aesthetics, Creativity, and the Arts. Advance online publication. https://doi.org/10.1037/aca0000573

'''

import pandas as pd
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine
import numpy as np

class MAD:
    '''
    Class for loading word vectors and calculating MAD scores.
    '''

    def __init__(self, space_file):
        '''
        Constructor that initializes semantic space and stores word embeddings. 
        '''
        self.word_embeddings = {}
        with open(space_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float32')
                self.word_embeddings[word] = embedding

        self.stop_words = set(stopwords.words('english'))

    def vectorize_response(self, sentence):
        '''
        Vectorizes the sentence to create a multiplicative model.
        '''
        word_vectors = []
        self.words = sentence.split()
        for word in self.words:
            if word in self.word_embeddings:
                word_vectors.append(self.word_embeddings[word])
        resp_vectors = np.array(word_vectors)
        return resp_vectors

    def remove_stop_words(self, sentence):
        '''
        Removes stop words from the text
        '''
        words = sentence.split()
        words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(words)

    def calculate_distance(self, sentence, word):
        '''
        Calculates the maximum absolute distance between the word and any word in the sentence.
        '''
        # Remove stop words from the sentence, vectorize, and get the word vector for the item word
        sentence = self.remove_stop_words(sentence)
        resp_vectors = self.vectorize_response(sentence)
        if word not in self.word_embeddings:
            raise ValueError(f'Item "{word}" not found in embeddings.')
        word_vector = self.word_embeddings[word]
        
        # Find the maximum cosine similarity between the word and any word in the sentence
        all_values = {sentence.split()[i]: cosine(resp_vectors[i], word_vector) for i in range(len(resp_vectors))}
        max_key = max(all_values, key=all_values.get)
        max_dist = np.max(list(all_values.values()))
        return max_dist, max_key

    def apply_to_dataframe(self, df):
        '''
        Apply the maximal distance calculation to a DataFrame.
        '''
        df[['maximal_distance', 'word_with_maximal_distance']] = df.apply(
            lambda row: pd.Series(self.calculate_distance(row['response'], row['item'])), axis=1
        )
        return df


if __name__ == '__main__':
    
    import pandas as pd
    import time
    
    start = time.time()
    
    glove_file = './MAD/spaces/glove.6B.300d.txt'
    glove_similarity = MAD(glove_file)

    example = {"item": ["apple", "apple","apple"], "response": ["eat apples sdfsd", "build a sculpture", "throw at a wall"]}
    df = pd.DataFrame(example)
    
    glove_similarity.apply_to_dataframe(df)
    
    stop = time.time()
    
    print(df)
    print(stop-start)
