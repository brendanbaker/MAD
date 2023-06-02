'''
MAD.py
 
Code for Maximum Absolute Distance (MAD) algorithm for creativity research
'''

import numpy as np
from scipy.spatial.distance import cosine
import nltk
from nltk.corpus import stopwords


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

    def vectorize_response(self, sentence):
        '''
        Vectorizes the sentence to create a multiplicative model.
        '''
        word_vectors = []
        words = sentence.split()
        for word in words:
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
        sentence = self.remove_stop_words(sentence)
        resp_vectors = self.vectorize_response(sentence)
        if word not in self.word_embeddings:
            raise ValueError(f'Word "{word}" not found in GloVe embeddings.')
        word_vector = self.word_embeddings[word]
        # Find the maximum cosine similarity between the word and any word in the sentence
        all_values = {sentence.split()[i]: cosine(resp_vectors[i], word_vector) for i in range(len(resp_vectors))}
        similarity = np.max(list(all_values.values()))
        return similarity, all_values, sentence


if __name__ == '__main__':
    
    glove_file = './MAD/spaces/glove.6B.300d.txt'
    glove_similarity = MAD(glove_file)

    sentence = 'will be sure to eat a banana and some apples'
    word = 'programming'

    similarity_score = glove_similarity.calculate_distance(sentence, word)
    print(f"Distance between '{sentence}' and '{word}': {similarity_score}")
