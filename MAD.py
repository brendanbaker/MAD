'''
MAD.py

Code for Maximum Absolute Distance (MAD) algorithm for creativity research
'''

import numpy as np
from scipy.spatial.distance import cosine
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


class MAD:
    '''
    Class for loading word vectors and calculating MAD scores.
    '''
    def __init__(self, space_file):
        '''
        Constructor that initializes semantic space.
        '''
        self.word_embeddings = {}
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float32')
                self.word_embeddings[word] = embedding

        self.stop_words = set(stopwords.words('english'))
    def vectorize_sentence(self, sentence):
        '''
        Vectorizes the sentence to create a multiplicative model.
        '''
        words = sentence.split()
        sentence_vector = np.ones(300)  # Assuming 300-dimensional GloVe vectors
        for word in words:
            if word in self.word_embeddings:
                sentence_vector *= self.word_embeddings[word]
        return sentence_vector
    def remove_stop_words(self, sentence):
        '''
        Removes stop words from the 
        '''
        words = sentence.split()
        words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(words)
    def calculate_distance(self, sentence, word):
        sentence = self.remove_stop_words(sentence)
        sentence_vector = self.vectorize_sentence(sentence)
        if word not in self.word_embeddings:
            raise ValueError(f'Word "{word}" not found in GloVe embeddings.')
        word_vector = self.word_embeddings[word]
        similarity = cosine(sentence_vector, word_vector)
        return similarity
        
        
if __name__ == '__main__':
    glove_file = './spaces/glove.6B.300d.txt'
    glove_similarity = MAD(glove_file)

    sentence = 'I love coding in Python'
    word = 'programming'

    similarity_score = glove_similarity.calculate_distance(sentence, word)
    print(f"Distance between '{sentence}' and '{word}': {similarity_score}")
        
        
