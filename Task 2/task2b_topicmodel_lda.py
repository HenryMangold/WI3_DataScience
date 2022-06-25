import sys
import pandas as pd
import re
import os
import xlsxwriter
from os import listdir
import pickle
from os.path import isfile
from os.path import join
from datetime import date
from wordcloud import WordCloud
from pprint import pprint

import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import gensim.corpora as corpora

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

import task2a_prep_functions as p

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# Script to train multiple LDA Models with different configurations and show differences in an Excel File #


def data_loading(source_path):
    data = pd.read_csv(source_path)
    return data


def data_cleaning(data):
    # add place name as title to beginning of content
    data["content"] = data["Place"] + '. ' + data["Content"]

    # remove the not needed columns from dataframe
    data = data.loc[:, ['content']].copy()

    # remove punctuation
    data['content_processed'] = data['content'].map(lambda x: re.sub('[,\.!?]', '', x))

    # convert text to lowercase
    data['content_processed'] = data['content_processed'].map(lambda x: x.lower())

    return data


def data_exploration(data):
    # join the content of all rows together to one huge string
    content_string = ','.join(list(data['content_processed'].values))

    # create a wordcloud object
    wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')

    # generate a word cloud based on the content string
    wordcloud.generate(content_string)

    # visualize the word cloud
    wordcloud.to_image()  # ToDo - save to directory


def data_preparation(data):
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'de', 'km', 'one', 'two'])

    def sent_to_words(sentences):
        for sentence in sentences:
            yield simple_preprocess(str(sentence), deacc=True)  # -> True removes punctuations

    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    # tokenizing and remove punctuation of processed content
    data_list = data.content_processed.values.tolist()
    data_words = list(sent_to_words(data_list))

    # lemmatization of processed content
    data_lemmatize = p.lemmatize_to_list([data_words])[0]

    # remove stop words of processed content
    data_words = remove_stopwords(data_lemmatize)

    # create Dictionary based on the document texts
    id2word = corpora.Dictionary(data_words)

    # create Corpus (Term Document Frequency) based on the document texts
    texts = data_words
    corpus = [id2word.doc2bow(text) for text in texts]

    return id2word, corpus, data_lemmatize


def train_model(corpus, id2word, num_topics, alpha, eta):
    # train model with corpus and dictionary of the document texts based on defined hyperparameters > Training with
    # lda multicore for using all CPU cores to parallelize and speed up model training. "An optimized implementation
    # of the LDA algorithm, able to harness the power of multicore CPUs. Follows the similar API 'lda-model'"
    if alpha and eta:  # > alpha & beta
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topics,
                                               alpha='symmetric',
                                               eta='symmetric')

    elif alpha:  # > alpha represents document-topic density
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topics,
                                               alpha='symmetric')

    elif eta:  # > beta represents topic-word density
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topics,
                                               eta='symmetric')
    else:  # > without Hyperparameters
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topics)

    # print the Keyword in the 10 topics
    pprint(lda_model.print_topics())

    return lda_model


def analyze_and_save_model(lda_model, corpus, id2word, num_topics, alpha, eta, data_lemmatize, model_name):
    # setup visualization and saving directory
    pyLDAvis.enable_notebook()

    dest_path = f'../results/{model_name}/'
    if not os.path.exists(dest_path):
        os.mkdir(f'../results/{model_name}/')
    LDAvis_data_filepath = os.path.join(f'../results/{model_name}/lda_vis_prepared_' + str(num_topics) + f'_alpha{alpha}_beta{eta}')

    # execute visualization prep yourself
    LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)

    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)

    # save visualization to html
    pyLDAvis.save_html(LDAvis_prepared, f'../results/{model_name}/lda_vis_prepared_' + str(num_topics) + f'_alpha{alpha}_beta{eta}.html')

    # show visualization results
    # LDAvis_prepared

    # "One approach to find optimum number of topics is: Build many LDA models with different values
    # of number of topics and pick the one that gives the highest coherence and lowest perplexity value."

    # compute Perplexity -> lower the perplexity better the model.
    perplexity = lda_model.log_perplexity(corpus)
    # print('\nPerplexity: ', perplexity)

    # compute Coherence score -> higher the topic coherence, the topic is more human interpretable.
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatize, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    # print('\nCoherence Score: ', coherence_lda)

    # save Perplexity and Coherence Score to lda model results directory
    with open(f'../results/{model_name}/lda_vis_prepared_' + str(num_topics) + f'_alpha{alpha}_beta{eta}_scores.txt', 'w') as f:
        f.write(f'Perplexity: {perplexity}\n')
        f.write(f'Coherence Score: {coherence_lda}\n')

    # calculate score for each topic per document
    def get_doc_topic(corpus, model):
        doc_topic = []
        for doc in corpus:
            doc_topic.append(model.__getitem__(doc, eps=0))
        return doc_topic

    matrix = get_doc_topic(corpus, lda_model)

    # reformat output to match matrix from LSA approach
    matrix_df = pd.DataFrame(matrix)
    matrix_df = matrix_df.apply(lambda x: [y[1] for y in x])

    # safe matrix as csv
    matrix_df.to_csv(f'../results/{model_name}/lda_vis_prepared_' + str(num_topics) +
                     f'_alpha{alpha}_beta{eta}_matrix.csv', index=False, header=False)


def predict(lda_model, text):
    #result_df = ...text...
    return


# Script to train multiple LDA Models with different configurations and show differences in an Excel File #
if __name__ == '__main__':

    print('Loading data')
    # Step 1: Loading Data #
    # read-in scraped data csv file from task 1
    csv_path = '../data/results_df_prep_all.csv'
    raw_data = data_loading(csv_path)
    print('Loading data done!')

    print('Cleaning data')
    # Step 2: Data Cleaning #
    # remove the not needed columns, remove punctuation, convert text to lowercase, ...
    prep_data = data_cleaning(raw_data)
    print('Cleaning data done!')

    # print('Exploration of data')
    # # Step 3: Data Exploration #
    # # create word cloud for exploration (optional)
    # data_exploration(prep_data)
    # print('Exploration of data done!')

    print('Preparation of data for LDA')
    # Step 4: Data Preparation (especially for LDA analysis) #
    # tokenizing and remove punctuation, lemmatization, remove stopwords, ...
    id2word, corpus, data_lemmatize = data_preparation(prep_data)
    print('Preparation of data done!')

    print("Training and analyzing of LDA Model's")
    # Step 5: LDA model training #
    # train model with different configurations to find best variant
    configurations = {0: [5, 'False', 'False'], 1: [5, 'True', 'False'], 2: [5, 'False', 'True'], 3: [5, 'True', 'True'],
                      4: [10, 'False', 'False'], 5: [10, 'True', 'False'], 6: [10, 'False', 'True'], 7: [10, 'True', 'True'],
                      8: [15, 'False', 'False'], 9: [15, 'True', 'False'], 10: [15, 'False', 'True'], 11: [15, 'True', 'True'],
                      12: [20, 'False', 'False'], 13: [20, 'True', 'False'], 14: [20, 'False', 'True'], 15: [20, 'True', 'True'],
                      16: [25, 'False', 'False'], 17: [25, 'True', 'False'], 18: [25, 'False', 'True'], 19: [25, 'True', 'True'],
                      20: [30, 'False', 'False'], 21: [30, 'True', 'False'], 22: [30, 'False', 'True'], 23: [30, 'True', 'True']}
    i_end = len(configurations)
    for i, key in enumerate(configurations):
        config = configurations[key]
        num_topics = config[0]
        alpha = eval(config[1])
        eta = eval(config[2])
        print(f'Performing configuration {(i+1)}/{i_end}  (topic n°: {num_topics} alpha: {alpha} eta: {eta})', end='\r',
              flush=True)

        # train lda model with specific configurations
        model = train_model(corpus, id2word, num_topics, alpha, eta)

        # defining model name for saving purpose
        today = date.today()
        d1 = today.strftime('%d%m%Y')
        model_name = f'{d1}_' + f'{num_topics}' + 'topics'

        # Step 6: Analyzing trained LDA model #
        # analyzing and saving trained model results
        analyze_and_save_model(model, corpus, id2word, num_topics, alpha, eta, data_lemmatize, model_name)

        # Step 7: Predicting with LDA model and test accuracy#
        # predict on trained model and save results
        #text = ...
        #df = predict(model, text)

        print('Finished!')
    print("Training and analyzing of Model's done!")

    print('Consolidate different model results into an Excel')
    # Step 8: Create Excel File for finding the best configuration variant #
    # merge the different results of configurations into an Excel
    workbook = xlsxwriter.Workbook('../results/merged.xlsx')
    worksheet = workbook.add_worksheet()
    # setup columns and their width's
    longest_width_B = 10
    longest_width_CD = 10
    text = 'Number of Topics'
    longest_width_A = len(text)
    cell_format = workbook.add_format({'bold': True})
    cell_format.set_align('center')
    worksheet.write('A1', 'Number of Topics', cell_format)
    worksheet.write('B1', 'Hyperparameter', cell_format)
    worksheet.write('C1', 'Perplexity', cell_format)
    worksheet.write('D1', 'Coherence Score', cell_format)

    # read directory's content and write to excel
    # iterate over directory's
    for base, dirs, files in os.walk('../results/'):
        file_counter = 1
        for directory in dirs:
            dict_counter = file_counter + 1
            worksheet.write(('A' + str(dict_counter)), directory.split('_')[1])  # Number of Topics
            # iterate over files in directory
            path = f'../results/{directory}/'
            files = [f for f in listdir(path) if isfile(join(path, f))]
            for file in files:
                if file.endswith('.html') or file.endswith('.txt') or file.endswith('.csv'):
                    None
                else:
                    file_counter += 1
                    text = file.split('_')[4] + file.split('_')[5]
                    width_B = len(text)
                    if width_B > longest_width_B:
                        longest_width_B = width_B
                    worksheet.write(('B' + str(file_counter)), text)  # Hyperparameter
                    txt_path = path + file + '_scores.txt'
                    with open(txt_path) as f:
                        lines = f.readlines()
                        f.close()
                    for i, line in enumerate(lines):
                        text = (line.split(': ')[1]).strip()
                        width_CD = len(text)
                        if width_CD > longest_width_CD:
                            longest_width_CD = width_CD
                        if i == 0:
                            worksheet.write(('C' + str(file_counter)), float(text))  # Perplexity
                        elif i == 1:
                            worksheet.write(('D' + str(file_counter)), float(text))  # Coherence Score
        # set column width
        worksheet.set_column(0, 0, longest_width_A)  # column Number of Topics
        worksheet.set_column(1, 1, longest_width_B)  # column Hyperparameter
        worksheet.set_column(2, 3, longest_width_CD)  # column Perplexity & Coherence Score

        # conditional formatting to find the best variant in excel file
        # > lower the perplexity better the model.
        # > higher the topic coherence, the topic is more human interpretable.
        worksheet.conditional_format(('C1:C' + str(file_counter)),
                                     {'type': '3_color_scale', 'min_color': '#88cc00', 'mid_color': '#ffcc00',
                                      'max_color': '#ff471a'})  # > Perplexity
        worksheet.conditional_format(('D1:D' + str(file_counter)),
                                     {'type': '3_color_scale', 'min_color': '#aaff00', 'mid_color': '#99e600',
                                      'max_color': '#88cc00'})  # > Coherence Score
    workbook.close()
    print('Consolidate model results into Excel done!')

    print('Script finished!')
    print(f'> You will find the Excel with the model results in following directory:  ../results/merged.xlsx')
    sys.exit()