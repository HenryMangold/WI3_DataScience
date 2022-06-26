import pandas as pd
import numpy as np
import re
import ast
from scipy import spatial

from gensim.models import LsiModel
from gensim import corpora
from gensim.utils import SaveLoad
from gensim.utils import simple_preprocess
import task2a_prep_functions as prep

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')



# load required data
# Topic model
lsa_model = LsiModel.load('models/lsa_model_test')
print(f'model object: {lsa_model}')
# load dt_matrix for model
dt_matrix = np.genfromtxt('output/lsa_example_matrix.csv', delimiter=',')
# dictionary of the topic model
dictionary = SaveLoad.load('dictionary/dict_test')
print(f'dictionary: {dictionary}')
# dataframe with original data from crawling
data = pd.read_csv('../data/results_df_prep.csv')


def preprocess_string(string):
    """run preprocessing steps on string and output a list of tokens"""
    def sent_to_words(sentence):
        yield simple_preprocess(str(sentence), deacc=True) # -> True removes punctuations

    def remove_stopwords(text):
        return [word for word in simple_preprocess(str(text)) if word not in stop_words]

    # Remove possible punctuation
    string = re.sub('[,\.!?]', '', string)
    # Convert text to lowercase
    string = string.lower()

    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'de', 'km', 'one', 'two'])

    # Tokenizing and remove punctuation of processed content
    string_words = list(sent_to_words(string))

    # Lemmatization of processed content
    string_lemmatize = prep.lemmatize_to_list([string_words])[0]

    # Remove stop words of processed content
    output_list = remove_stopwords(string_lemmatize)

    return output_list


def find_dream_destination(input):
    """find destinations that best match the given input string based on topic modeling and clustering"""
    # generate scoring dataframe for output
    destinations = pd.DataFrame(columns=['place', 'topic_score', 'no_ne_count', 'no_ne_score', 'total_score'])
    destinations['place'] = data['Place']

    # CALCULATE TOPIC MODELING SCORES
    # generate tree from document topic matrix of model to later search for closest document vector to input vector
    tree = spatial.KDTree(dt_matrix)

    # Preprocessing of input string
    input_prep = preprocess_string(input)
    print(input_prep)

    # transform words with dict to bow
    input_bow = dictionary.doc2bow(input_prep)

    # generate topic vector for inputs and reformat
    input_topics = lsa_model.__getitem__(input_bow)
    input_vec = [y[1] for y in input_topics]

    # get closest document vector for each word vector
    distance, place_id = tree.query(input_vec, k=100)
    # write position of place into dataframe
    for i, (distance, place) in enumerate(zip(distance, place_id)):
        destinations['topic_score'][place] = i + 1

    # CALCULATE NO-NE SCORES
    # extract No-NEs from input string
    _void, no_ne_list = prep.get_hypernyms([input_prep])
    print(no_ne_list)
    # search for no-ne attractions in documents and get a count for them
    no_ne_count = [0] * 100
    for no_nes in no_ne_list:
        for no_ne in no_nes:
            for i, row in enumerate(data['no_NE_attractions_plain_unique']):
                if no_ne in ast.literal_eval(row):
                    no_ne_count[i] += 1
    destinations['no_ne_count'] = no_ne_count
    # sort dataframe for no_ne_count and write scores
    destinations.sort_values(by=['no_ne_count', 'topic_score'], ascending=[False, True], inplace=True)
    postions = range(1, 101, 1)
    destinations['no_ne_score'] = postions


    # calculate total score based on topic and NE clustering score
    destinations.fillna(0, inplace=True)
    destinations['total_score'] = destinations['topic_score'] + destinations['no_ne_score']
    # lower scores are better, so we sort the dataframe accordingly before returning
    destinations.sort_values(by='total_score', inplace=True)

    return destinations


if __name__ == "__main__":
    # example input string:
    # 'I want to go somewhere with lots of beaches and warm weather. The place should have a long coastline.'
    string = str(input('type text for destination finder'))
    destinations = find_dream_destination(string)
    print(destinations.head())
