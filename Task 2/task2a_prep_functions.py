######################################### IMPORTS AND START #####################################################


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
import spacy
from spacy.tokens import Span
from spellchecker import SpellChecker


#Download ressources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')
    

######################################### PREPROCESSING #####################################################

def label_continent (row):
    ''' Get continents for countries for row in column

        Args:
            row: Row in column with countries

        Returns:
            String -> String with continent
        '''

    if row['Country'] == 'brazil' :
        return 'America'
    if row['Country'] == 'canada' :
        return 'America'
    if row['Country'] == 'france' :
        return 'Europe'
    if row['Country'] == 'germany':
        return 'Europe'
    if row['Country']  == 'italy':
        return 'Europe/AA'
    if row['Country'] == 'mexico':
        return 'America.'
    if row['Country'] == 'peru':
        return 'America'
    if row['Country'] == 'spain':
        return 'Europe'
    if row['Country'] == 'sweden':
        return 'Europe'
    if row['Country'] == 'usa':
        return 'America'

    return 'Other'


def s_tokenize(column):
    ''' Convert text into sentence tokens
    
    Args:
        column: Text column to convert to sentence tokens
        
    Returns:
        List -> List with converted sentence tokens
    '''
    
    token_column = [sent_tokenize(text) for text in column]
        
    return token_column


def w_full_tokenize(column):
    ''' Convert text with sentences into tokens
    
    Args:
        column: Text column to convert to tokens
        
    Returns:
        List -> List with converted tokens
    '''
    
    token_column = [word_tokenize(text) for text in column]
        
    return token_column


def w_tokenize(column):
    ''' Convert text with sentences into tokens
    
    Args:
        column: Text column to convert to tokens
        
    Returns:
        List -> List with converted tokens
    '''
    
    token_column = []
    
    for text in column:
        token_column.append([word_tokenize(sent) for sent in text])
        
    return token_column


def w_column_tokenize(column):
    ''' Convert text with sentences into tokens
    
    Args:
        column: Text column to convert to tokens
        
    Returns:
        List -> List with converted tokens
    '''
    
    text_column = []
    
    for text in column:
        token_column = []

        for word in text:
            token_column.append([word_tokenize(sent) for sent in word])

        text_column.append(token_column)

    return token_column


def detokenize_w_to_s(column):
    ''' Convert tokens into sentences
    
    Args:
        column: Token column to convert to sentence
        
    Returns:
        List -> List with converted sentences
    '''
    
    text_column = []
    
    for text in column:
        sentences = []
        for sentence_list in text:
            sentence = ''
            for token in sentence_list:
                if sentence == '':
                    sentence = sentence + token
                else:
                    sentence = sentence + ' ' + token
                    
            sentences.append(sentence)   
                
        text_column.append(sentences)
        
    return text_column


def remove_list_depth_after_tokenisation(column):
    ''' Reduce the list depth of a column by 1 level
    
    Args:
        column: Column whose list depth is to be changed
                
    Returns:
        List -> List with converted tokens
    '''
    
    token_column = []
    
    for line in column:
        tokens_list = []
        
        for token_list in line:
            
            for token in token_list:
                tokens_list.append(token)   
                
        token_column.append(tokens_list)
        
    return token_column


def lower_tokens_to_list(column):
    '''
    Normalizes tokens to lowercase and returns result as a list
    
    Args:
        column: Column to bring tokens to lowercase
        
    Returns:
        List -> Column with lowercase tokens
    '''
    
    lower = []
    for text in column:
        list_low = []
        for sentence in text:
            lower_list = []
            for token in sentence:
                lower_list.append(token.lower())
            list_low.append(lower_list)
        lower.append(list_low)
    return lower


def keep_only_words(column):
    ''' Remove any punctuation marks and numbers from column
    
    Args:
        column: Column to remove values from    
        
    Returns:
        List -> Column with omly words
    '''
    text_column = []
    
    for text in column:
        text_in_line = []
        
        for sentence in text:
            sent_new = []
            for token in sentence:
                trans_token = token.translate(str.maketrans('', '', string.punctuation))
                if trans_token == '':
                    continue

                elif trans_token.isalpha():
                    sent_new.append(token)

            text_in_line.append(sent_new)

        text_column.append(text_in_line)

    return text_column


def spelling_correction_suggestions(list_of_1grams, returning = False):
    ''' Remove common spelling misstakes from collumn
    
    Args:
        list_of_1grams: List to find spelling misstakes from  
        returning: Indicates if suggestions should be printed or returned
        
    Returns:
        Dict -> Dict with misspelled tokens
    '''
    
    spell = SpellChecker()

    # find those words that may be misspelled
    misspelled = spell.unknown(list_of_1grams)
    misspelled_dict = dict()

    for word in misspelled:
        if returning:
            misspelled_dict[word] = spell.correction(word)
            
        else:
            # Get the one `most likely` answer
            print(f'Old word: {word}')

            print(f'New word: {spell.correction(word)}')

            # Get a list of `likely` options
            print('Candidates',spell.candidates(word))
            print()
    
    if returning:
        return misspelled_dict

        
def spelling_correction(column, dict_words_to_correct):
    ''' Remove common spelling misstakes from colSumn
    
    Args:
        column: Column to remove spelling misstakes from  
        dict_words_to_correct: Dictonary with words which should be corrected:
                {right_spelling: [wrong_spelling_1, wrong_spelling_2, ...]}
        
    Returns:
        List -> Column without spelling misstakes
    '''
    
    new_column = []
    
    for line in column:
        words_in_line = []
        
        for token in line:
            word_new = token
            
            for key in dict_words_to_correct:
                
                for wrong_spelling in dict_words_to_correct[key]:
                    
                    replace_string = wrong_spelling
                    word_new = word_new.replace(replace_string,key)

                    replace_string = wrong_spelling.lower()
                    word_new = word_new.replace(replace_string,key.lower())
                    
                    replace_string = wrong_spelling.upper()
                    word_new = word_new.replace(replace_string,key.upper())

                    replace_string = wrong_spelling.capitalize()
                    word_new = word_new.replace(replace_string,key.capitalize())
                
            words_in_line.append(word_new)
            
        new_column.append(words_in_line)
        
    return new_column      


def token_and_tag_to_list(column):
    '''Find word tags for tokens and safe them in list
    
    Args:
        column: Column to find pos tags for
        
    Returns:
        List -> List with assigned POS tags
    '''
    
    #use nltk.pos_tag to create a list called "upper_tokens_pos"
    upper_tokens_pos_list = []
    for text in column:
        token_list = []
        for sentence in text:
            try:
                token_list.append(nltk.pos_tag(sentence))
            except LookupError:
                nltk.download('averaged_perceptron_tagger')
                token_list.append(nltk.pos_tag(sentence))
                
        upper_tokens_pos_list.append(token_list)
    
    #concatenate upper tokens and the assigned POS tag to a new list "token_and_tag" (return value)
    return upper_tokens_pos_list


def nltk_tag_to_wordnet_tag(nltk_tag):
    ''' Function to convert nltk tag to wordnet tag
    
    Args:
        nltk_tag: Wordnet for text
        
    Returns:
        wordnet
    '''
    try:
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:          
            return None
    
    except LookupError:
        nltk.download('omw-1.4')
        nltk.download('wordnet')
        nltk_tag_to_wordnet_tag(nltk_tag)


def lemmatize_to_list(column):
    ''' Lemmatize the column tokens
    
    Args:
        column: Column to lemmatize        
        
    Returns:
        List -> Column with lemmatized tokens
    '''
    
    lemmatizer = WordNetLemmatizer()
    lemma_list = []
    for text in column:
        token_list = []
        for sentence in text:
            #tokenize the sentence and find the POS tag for each token using nltk.pos_tag
            try:
                nltk_tagged = nltk.pos_tag(sentence)
            except LookupError:
                nltk.download('averaged_perceptron_tagger')
                nltk_tagged = nltk.pos_tag(sentence)
                
            #tuple of (token, wordnet_tag)
            wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)

            lemmatized_sentence = []
            for word, tag in wordnet_tagged:
                if tag is None:
                    #if there is no available tag, append the token as is
                    lemmatized_sentence.append(word)
                else:        
                    #else use the tag to lemmatize the token
                    lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
            token_list.append(lemmatized_sentence)
        lemma_list.append(token_list)
    return lemma_list


def stemmed_token_to_list(column):
    ''' Funktion to stemm column tokens
    
    Args:
       column: Column with tokens to stemm
        
    Returns:
       List -> Column with stemmed tokens
    '''
    
    #Intialize porter stemmer object
    ps = PorterStemmer()
    #Initialize empty list for stemmed_tokens
    stemmed_tokens=[]
    #Iterate tokens in 'Upper Token'
    for text in column:
        stemmed_list = []
        for sentence in text:
            stemmed = []
            for token in sentence:
                stemmed.append(ps.stem(token))
            stemmed_list.append(stemmed)
        stemmed_tokens.append(stemmed_list)
        
    return stemmed_tokens


def remove_stopwords(column):
    ''' Remove common englisch stopwords from column
    
    Args:
        column: Column to remove stopwords from  
        
    Returns:
        List -> Column without stopwords
    '''
    
    #Create a set holding english stopwords using the stopwords class
    try:
        stop_words=set(stopwords.words("english"))
    
    except LookupError:
        nltk.download('stopwords')
        stop_words=set(stopwords.words("english"))
        
    new_column = []
    
    for text in column:
        liste_text = []
        
        for sentence in text:
            liste_sentence = []

            for token in sentence:
            
                #iterate tokens in series, if stop wordnot append, else keep token
                if token not in stop_words:
                    liste_sentence.append(token)

            if liste_sentence != []:
                liste_text.append(liste_sentence)
                    
        new_column.append(liste_text)
        
    return new_column


def frequency_detection(column):
    """ Detect frequency in token list
    
    Args:
        column: Column to remove detect frequency from  
        
    Returns:
        List -> Column with frequencies
    """

    new_column = []

    for text in column:
        text_column = []

        fdist = FreqDist(text)

        for i in fdist.items():
            text_column.append(i)

        new_column.append(text_column)

    return new_column


def tokens_to_string(column):
    """ Transform token list to string representation

    Args:
        column: Column with tokens

    Returns:
        List -> List with strings
    """
    new_column = []

    for text in column:
        new_column.append(' '.join(text))

    return new_column


def one_hot_encoding(df, column_name):
    '''Encoding of data frame

    Args:
        df: dataframe to encode
        column_name: Name of column with string tokens

    Returns:
        DataFrame -> encoded dataframe
    '''

    # Create the numericalizer TFIDF for lowercase
    tfidf = TfidfVectorizer(decode_error='ignore', lowercase=True, min_df=2)

    # Numericalize the train dataset
    train_tf = tfidf.fit_transform(df[column_name].values.astype('U'))
    vocabulary = tfidf.get_feature_names()

    # create occurency frequency matrix
    corpus = df[column_name]
    pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)), ('tfidf', TfidfTransformer())]).fit(corpus)
    data_idf = pipe['count'].transform(corpus).toarray()

    df_idf = pd.DataFrame(data=data_idf, columns=vocabulary, index=df.index)
    count_docs = len(df_idf)
    row = []
    for col in df_idf.columns:
        row.append(df_idf[col].sum())
    df_idf.loc[count_docs + 1] = row

    return df_idf


def transpose_df(df):
    '''Transposing dataframe

    Args:
        df: Frame to transpose

    Returns:
        df_idf_t: Transposed frame
        count_docs: Count of documents
    '''

    count_docs = len(df)
    df_idf_t = df.transpose()
    df_idf_t = df_idf_t.rename(columns={count_docs: 'count_tokens_overall'})
    row = []
    for col in df_idf_t.columns:
        row.append(df_idf_t[col].sum())
    df_idf_t.loc['count_tokens_in_doc'] = row
    return df_idf_t, count_docs


def create_tfidf_matrix(df, count_docs):
    '''Creation of TFIDF matrix

    Args:
        df: Input dataframe
        count_docs: Count of documents

    Returns:
        df: Matrix
    '''

    df.columns = df.columns.map(str)
    df = df.add_prefix('occ_')

    count_all_tokens = len(df.index)
    n = count_docs
    dfi = df['occ_count_tokens_overall']
    quotient = n / dfi

    for col in df.columns:
        if col != 'occ_count_tokens_overall':
            # calculate tf
            freq_col_name = 'freq_' + col[4:]
            count_tokens_in_document = df[col][count_all_tokens - 1]
            tf = df[col] / count_tokens_in_document
            df[freq_col_name] = tf

            # calculate tf-idf weights
            tf_idf_col_name = 'tf_idf_' + col[4:]
            log = np.log10(quotient)
            tf_idf_weight = log * tf
            df[tf_idf_col_name] = tf_idf_weight

    return df


######################################### DETERMINATION NAMED ENTETIES #####################################################


def named_entity_recognition_nltk(column):
    ''' Find named entities in column with nltk
    
    Args:
        column: Column to find entities in
        
    Returns:
        List -> List with entities
    '''
    text_column = []

    for text in column:

        text_list = []

        for sentence in text:

            try:
                pos_tag = nltk.pos_tag(word_tokenize(sentence))

                chunk = nltk.ne_chunk(pos_tag)

                NE = [ " ".join(w for w, t in ele).lower() for ele in chunk if isinstance(ele, nltk.Tree)]
                
                text_list.append(NE)

            except LookupError:
                nltk.download('maxent_ne_chunker')
                nltk.download('words')

        text_column.append(text_list)

    return text_column

                           
def named_entity_recognition_spacy(column, dict_of_words = None):
    ''' Find named entities in column with spacy

    Args:
        column: Column to find entities in
        dict_of_words: Additional words to add to findings

    Returns:
        List -> List with entities
    '''

    global new_ent
    text_column = []
    text_column_plain = []
    text_column2 = []

    nlp = spacy.load("en_core_web_trf")

    set_uniques = set()

    for text in column:

        text_list = []
        text_list_plain = []
        text_list2 = []

        for sentence in text:

            doc = nlp(sentence)
            new_ent = None
            if dict_of_words != None:

                for key, val in dict_of_words.items():
                    if key in sentence:
                        try:
                            # get hash value of FAC entity label
                            label_hash = doc.vocab.strings[val]

                            # create a span for new entity
                            for sent in doc.sents:
                                for token in sent:
                                    if key == token.text:
                                        new_ent = Span(doc, token.i - sent.start, token.i - sent.start+1, label=label_hash)

                            # Add entity to existing Doc object
                            doc.ents = list(doc.ents) + [new_ent]

                        except ValueError:
                            continue

            sentence_list = []
            sentence_list_plain = []
            sentence_list2 = []
            for t in doc.ents:

                if t.text.lower() in ['de', 'la', 'T', 'and', 'St', 'Po', '-', 'So', 'Pl', 'Fo']:
                    continue

                sentence_list2.append(t.text.lower())
                if t.label_ != 'QUANTITY' and t.label_ != 'DATE' and t.label_ != 'TIME' and t.label_ != 'PERCENT' and t.label_ != 'LANGUAGE' and t.label_ != 'PEOPLE' and t.label_ != 'LAW' and t.label_ != 'MONEY' and t.label_ != 'NORP' and t.label_ != 'CARDINAL' and t.label_ != 'ORDINAL' and t.label_ != 'WORK_OF_ART' and t.label_ != 'PERSON':
                    sentence_list.append((t.text.lower(), t.label_))
                    sentence_list_plain.append(t.text)
                    set_uniques.add(t.label_)

            text_list.append(sentence_list)
            text_list_plain.append(sentence_list_plain)
            text_list2.append(sentence_list2)

        text_column.append(text_list)
        text_column_plain.append(text_list_plain)
        text_column2.append(text_list2)

    return text_column,text_column_plain, text_column2


def filter_tokens_start_caps(column):
    ''' Identify all tokens with capital start
    
    Args:
        column: column with sentences
        
    Returns:
        List -> Column with tokens
    '''

    new_column = []

    for text in column:
        text_list = []

        for sentence in text:
            sentence_list = []

            for token in sentence:

                if token.istitle():

                    sentence_list.append(token)

            text_list.append(sentence_list)

        new_column.append(text_list)

    return new_column


def ne_ident_tokens(df):
    ''' Identify all NE's per text
    
    Args:
        df: Dataframe containing NE columns 
        
    Returns:
        List -> Column with strings
    '''
    
    new_column = []
    text_list = []

    for index, row in df.iterrows():
        
        string_set = set()

        for liste in row[0]:

            for ele in liste:
                
                string_set.add(ele)

        for liste in row[1]:

            for ele in liste:

                string_set.add(ele)

        text_list.append(list(string_set))

    new_column.append(text_list)
        
    return new_column


def unique_and_dict_count(column):
    new_column = []
    new_column2 = []
    

    for text in column:

        text_list = list(set(text))
        text_dict = dict()

        for i in text_list:
            text_dict[i] = 0

        for token in text:
            text_dict[token] += 1

        new_column.append(text_list)
        new_column2.append(text_dict)

    return new_column, new_column2


def detect_NE_missing(column_propbly_NE, column_NE):
    """ Detection of missing NE's

    Args:
        column_propably_NE: Column to find additional NEs in
        column_NE: Column with detected NEs

    Returns:
        List -> List with additional NE's
    """
    new_column = []

    for prop, real in zip(column_propbly_NE, column_NE):

        text_column = []

        diff = set(prop).difference(set(real))

        if diff != set():

            for ele in diff:
                text_column.append(ele)

        new_column.append(text_column)

    return new_column
              

######################################### DETERMINATION NOT NAMED ENTETIES #####################################################


def keep_nouns(column):
    '''Remove everything aside from nouns automaticly from column
    
    Args:
        column: Column to remove everything aside from nouns from
        
    Returns:
        nouns_list: Cleaned list
    '''
    
    #use nltk.pos_tag to create a list called "upper_tokens_pos"
    upper_tokens_pos_list = []
    for text in column:
        text_list = []
        for sentence in text:
            sentence_list = []
            sentence_pos = []
            for token in sentence:
                try:
                    sentence_pos.append(nltk.pos_tag([token])[0])
                except LookupError:
                    nltk.download('averaged_perceptron_tagger')
                    sentence_pos.append(nltk.pos_tag([token])[0])

            for token in sentence_pos:
                if token[1] in ['NN', 'NNS', 'NNP', 'NNPS']:
                    sentence_list.append(token[0])
            text_list.append(sentence_list)
        upper_tokens_pos_list.append(text_list)
    
    return upper_tokens_pos_list


def remove_words_from_sentences(df):
    ''' Remove word in list from sentences
    
    Args:
        df: Dataframe to remove names from    
        
    Returns:
        List -> Column with strings
    '''
    text_list = []

    for index, row in df.iterrows():
        sentence_list = []

        for word in row[0]:
            
            found = False

            for ne in row[1]:

                for token in ne:
                
                    if token == word:
                        found = True
            if found == False:
                sentence_list.append(word)

        text_list.append(sentence_list)
        
    return text_list


def get_hypernyms(column):
    '''Find all not-NE nouns which are tourist attractions
    
    Args:
        column: Column to find not-NE attractions in
        
    Returns:
        list -> List with found not-NE nouns
    '''

    list_of_tourist_categories = [['point_of_view'],['buying'], ['entertainment'], ['party'],['resort_area'], ['building_complex'],['location', 'space'],['location', 'outline'], ['jungle'],['vantage'], ['geographic_point'], ['animal'], ['passageway'],['cavity'],['defensive_structure'],['sport'], ['body_of_water'], ['urban_area'],['approach'],['tract'], ['country'], ['wilderness'], ['biome'], ['land'], ['headwater'], ['topographic_point'], ['geological_formation'], ['food'], ['vegetation'], ['building'], ['structure', 'bridge'], ['place_of_business'], ['ship']]

    new_column = []
    new_column_plain = []

    for text in column:
        token_list = []
        token_list_plain = []

        for sentence in text:

            synsets_list = wordnet.synsets(sentence)

            if synsets_list != [] and not sentence in ['home', 'place', 'site', 'spot', 'stretch', 'food', 'de', 'la']:
                synset_all_list = synsets_list[0].hypernym_paths()
                synset_flat_list = []

                for x in synset_all_list:
                    synset_flat_list = synset_flat_list + x

                synset_flat_list_names = []
                for i in synset_flat_list:
                    try:
                        name, type, pos = i.name().split('.')
                    except ValueError:
                        print(i)
                        continue

                    if type != 'n':
                        synset_flat_list_names = []
                        break
                    else: 
                        synset_flat_list_names.append(name)

                if synset_flat_list_names != []:

                    for j in list_of_tourist_categories:

                        if all(item in synset_flat_list_names for item in j):
                            
                            token_list_plain.append(sentence)
                            if j in [['animal']] :
                                token_list.append((sentence, 'animal_attraction'))

                            elif j in [['food']]:
                                token_list.append((sentence, 'culinary_attraction'))

                            elif j in [['passageway'],['defensive_structure'],['urban_area'],['building_complex'],['district'],['building'],['structure', 'bridge'],['topographic_point']]:
                                token_list.append((sentence, 'architectural_attraction'))

                            elif j in [['jungle'],['geographic_point'],['body_of_water'],['country'],['wilderness'],['biome'],['land'],['geological_formation'], ['vegetation'],['headwater'],['cavity'],['location', 'outline'],['location', 'space']]:
                                token_list.append((sentence, 'geographic_attraction'))

                            elif j in [['place_of_business'],['ship'],['sport'],['party'],['buying'],['entertainment'],['resort_area']]:
                                token_list.append((sentence, 'entertainment_attraction'))

                            else:
                                token_list.append((sentence, 'other_attraction'))
                            break

        new_column_plain.append(token_list_plain)
        new_column.append(token_list)

    return new_column, new_column_plain


def transaction_encoding(column):
    ''' Function is encoding the column to a transaction matrix
    
    Args:
        column: Column with transactions
        
    Returns:
        df: Data Frame with encoded transactions
    '''
    try:
        transactions = column.tolist()
    except:
        transactions = column
        
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    return df


def frequent_itemsets_apriori(df_encoded, min_support, length=2, support=0.01):
    '''Creats frequent itemsets based on apriori algorithm
    
    Args:
       df_encoded: Encoded dataframe
       min_support: Minimum threshold support for set
       length: Minimum length of sets for filter
       support: Minimum threshold support for filter
       
    Returns:
       Dataframe -> With frequent itemsets        
    '''
    
    #Filter by number of items and support
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    #Create new column for itemset
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

    #Itemsets of length 2 that have a support of at least 1 percent
    frequent_itemsets = frequent_itemsets[ (frequent_itemsets['length'] == length) &
                       (frequent_itemsets['support'] >= support) ]
    
    return frequent_itemsets

    
def frequent_itemsets_fpgrowth(df_encoded, min_support, length=2, support=0.01):
    '''Creats frequent itemsets based on fpgrowth algorithm
    
    Args:
       df_encoded: Encoded dataframe
       min_support: Minimum threshold support for set
       length: Minimum length of sets for filter
       support: Minimum threshold support for filter
       
    Returns:
       Dataframe -> With frequent itemsets     
    '''
    
    #Filter by number of items and support
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    #Create new column for itemset
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

    #Itemsets of length 2 that have a support of at least 1 percent
    frequent_itemsets = frequent_itemsets[ (frequent_itemsets['length'] == length) &
                       (frequent_itemsets['support'] >= support) ]
    
    return frequent_itemsets


def association_rules(df_frequent_itemsets, metric="lift", min_threshold=2):
    '''Creation of association rules
    
    Args:
      df_frequent_itemsets: Dataframe with frequent itemsets
      metric: Metric for validation
      min_threshold: Min threshold of metric
        
    Returns:
      rules: Dataframe with association rules  
    '''
    
    rules = association_rules(df_frequent_itemsets, metric=metric, min_threshold=min_threshold)
    return rules


######################################### PREPROCESSING CLUSTERING #########################################


def create_n_grams_to_list(column, length):
    """ Create n-grams from tokens

    Args:
        column: Column to transform
        length: Length of the n-grams builded

    Returns:
        n_grams: Column with n-grams
    """

    n_grams_list = []

    for line in column:
        n_grams = []

        for token_list in line:
            grams = list(ngrams(token_list, length))
            if grams:
                n_grams.append(grams)

        n_grams_list.append(n_grams)

    return n_grams_list


def dataframe_for_training(encoded_frame, *args):
    """ Shapes the ouput to a data frame for training classification

    Args:
        encoded_frame: Frame to bring to training shape
        *args: Columns which should be appended as extra information

    Returns:
        df_merged: Dataframe ready for training
    """

    df_merged = encoded_frame

    for arg in args:
        df_merged = df_merged.join(arg)

    return df_merged


def list_to_string(column):
    """ Convert list to string

    Args:
        df_column: Dataframe with column

    Returns:
        string for every transaction in list
    """

    return [','.join(map(str, l)) for l in column]


def use_tfidfVectorizer(column):
    """ Usage of tfidf vektor to transform dataframe

    Args:
       df: Dataframe to transform

    Returns:
        train_tf: Transformed dataframe
    """

    # Create the numericalizer TFIDF for lowercase
    tfidf = TfidfVectorizer(decode_error='ignore', lowercase=True, min_df=2)

    # Numericalize the train dataset
    tfidf_list = tfidf.fit_transform(column.values.astype('U'))

    # Build data frame
    vocabulary = tfidf.vocabulary_
    vocabulary_transposed = {y: x for x, y in vocabulary.items()}
    train_tf = pd.DataFrame(tfidf_list.toarray())
    train_tf = train_tf.rename(columns=vocabulary_transposed)

    return train_tf


######################################### BONUS #########################################


def automatic_mapping_NE_notNE(df):
    """ Automatic detetcion of NE and not NE
    
    Args:
        df: DataFrame to detect in
        
    Returns:
        List -> Detections

    """

    new_column = []

    for index, row in df.iterrows():
        ne_list = []
        i = 0

        for sentence in row[1]:

            for ne in sentence:
                tokens = word_tokenize(ne)
                try:
                    while True:
                        #print(tokens[0].lower(), row[0][i])
                        check = tokens[0].lower()

                        if check == 'the':
                            check = tokens[1].lower()

                        if check == row[0][i]:
                            break
                        else:
                            i+=1

                    while True:
                        i+=1
                        if row[0][i] in row[2]:
                            ne_list.append((ne, row[0][i]))
                            #print(row[2])
                            #print(ne, row[0][i])
                            break
                except:
                    continue

        new_column.append(ne_list)

    return new_column
