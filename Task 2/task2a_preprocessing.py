import task2a_prep_functions as p
import pandas as pd


class Preprocessing:
    """ Class for generating the non-NE tourist attractions training data from raw data

        Attributes:
            __df_raw: DataFrame containing the raw data
            __df_mapping: DataFrame containing the mapping data
            __df: DataFrame containing the processed data

        Methods:
            __read_data(input_dir_raw_data)
                Reading raw data from the scraping process

            __mapping_country_continent()
                Adding the country and continent information to a special mapping DataFrame

            __preprocess_raw_data_lemma()
                Preprocessing of raw data including lemmatization

            __preprocess_raw_data_stemming()
                Preprocessing of raw data including stemming

            __create_no_NE_tourist_attractions()
                Generation of the NE and non NE tourist attractions

            __create_one_grams()
                Generation of one gram tokens

            __create_two_grams()
                Generation of two gram tokens

            __create_mixed_grams()
                Generation of mix of one and two gram tokens

            __create_true_false_matrix()
                Generation of a True False matrix for the tokens in the documents

            __create_tfidf_matrix()
                Generation of a TFIDF matrix for the tokens in the documents

            preprocess_data(input_dir, wordreduction)
                Reading and preprocessing of the data

            create_train_data(grams, train_matrix)
                Transformation of the preprocessed data into a training data matrix
    """

    def __int__(self):
        """ Initialization of the object variables

        Returns:
            None

        """

        self.__df_raw = None
        self.__df_mapping = None
        self.__df = None


    @property
    def df_mapping(self) -> pd.DataFrame:
        """ Return of the mapping data frame

        Returns:
            DataFrame -> DataFrame containing the mapping for place to country, continent, available NoNE tourist
            attractions and their categories

        """

        return self.__df_mapping


    def __read_data(self, input_dir_raw_data:str):
        """ Reading raw data from the scraping process

        Args:
            input_dir_raw_data: Path to the raw data

        Returns:
            None

        """

        self.__df_raw = pd.read_csv(input_dir_raw_data, index_col=0)


    def __mapping_country_continent(self):
        """ Adding the country and continent information to a special mapping DataFrame

        Returns:
            None

        """

        self.__df_mapping = self.__df_raw.copy()
        self.__df_mapping['Country'] = [i[3] for i in self.__df_mapping['Link'].str.split('/')]
        self.__df_mapping['Continent'] = self.__df_mapping.apply(lambda row: p.label_continent(row), axis=1)


    def __preprocess_raw_data_lemma(self):
        """ Preprocessing of raw data including lemmatization

        Returns:
            None

        """

        self.__df = self.__df_raw.copy()
        self.__df['sentences'] = p.s_tokenize(self.__df['Content'])
        self.__df['tokens_full'] = p.w_full_tokenize(self.__df['Content'])
        self.__df['tokens_full_lower'] = p.lower_tokens_to_list([self.__df['tokens_full'].tolist()])[0]
        self.__df['tokens_full_lower_digi'] = p.keep_only_words([self.__df['tokens_full_lower'].tolist()])[0]
        self.__df['lemmas'] = p.lemmatize_to_list([self.__df['tokens_full_lower_digi']])[0]
        self.__df['preprocessed'] = p.remove_stopwords([self.__df['lemmas']])[0]


    def __preprocess_raw_data_stemming(self):
        """ Preprocessing of raw data including stemming

        Returns:
            None

        """

        self.__df = self.__df_raw.copy()
        self.__df['sentences'] = p.s_tokenize(self.__df['Content'])
        self.__df['tokens_full'] = p.w_full_tokenize(self.__df['Content'])
        self.__df['tokens_full_lower'] = p.lower_tokens_to_list([self.__df['tokens_full'].tolist()])[0]
        self.__df['tokens_full_lower_digi'] = p.keep_only_words([self.__df['tokens_full_lower'].tolist()])[0]
        self.__df['stem'] = p.stemmed_token_to_list([self.__df['tokens_full_lower_digi']])[0]
        self.__df['preprocessed'] = p.remove_stopwords([self.__df['stem']])[0]


    def __create_no_NE_tourist_attractions(self):
        """ Generation of the NE and non NE tourist attractions

        Returns:
            None

        """

        self.__df['None'] = [[[]] for i in range(100)]
        small, small_plain, full = p.named_entity_recognition_spacy(self.__df['sentences'])
        self.__df['named_entities_spacy_small'] = small
        self.__df['named_entities_spacy_small_plain'] = small_plain
        self.__df['named_entities_spacy_full'] = full
        self.__df['NE_preprocessed'] = p.remove_stopwords(p.lemmatize_to_list(p.keep_nouns(p.keep_only_words(
            p.w_column_tokenize(p.ne_ident_tokens(self.__df[['None', 'named_entities_spacy_full']]))))))
        self.__df['sent_prepcorcessed_only_nouns'] = p.keep_nouns([self.__df['preprocessed']])[0]
        self.__df['sent_without_NE'] = p.remove_words_from_sentences(self.__df[['sent_prepcorcessed_only_nouns', 'NE_preprocessed']])
        kombs, plain = p.get_hypernyms(self.__df['sent_without_NE'])
        self.__df_mapping['no_NE_attractions'] = kombs
        self.__df_mapping['no_NE_attractions_plain'] = plain
        self.__df['no_NE_attractions_plain'] = plain
        unique, unique_dict = p.unique_and_dict_count(self.__df['no_NE_attractions_plain'])
        self.__df['no_NE_attractions_plain_unique'] = unique
        self.__df['no_NE_attractions_plain_unique_dict'] = unique_dict


    def __create_one_grams(self):
        """ Generation of one gram tokens

        Returns:
            None

        """

        self.__df['grams'] = self.__df['no_NE_attractions_plain_unique_dict']


    def __create_two_grams(self):
        """ Generation of two gram tokens

        Returns:
            None

        """

        self.__df['grams'] = p.create_n_grams_to_list(self.__df['no_NE_attractions_plain_unique_dict'], 2)


    def __create_mixed_grams(self):
        """ Generation of mix of one and two gram tokens

        Returns:
            None

        """

        self.__df['grams'] = self.__df['no_NE_attractions_plain_unique_dict']+p.create_n_grams_to_list(self.__df['no_NE_attractions_plain_unique_dict'], 2)


    def __create_true_false_matrix(self):
        """ Generation of a True False matrix for the tokens in the documents

        Returns:
            None

        """

        df_matrix = p.transaction_encoding(self.__df['grams'].tolist())
        df_matrix = p.dataframe_for_training(df_matrix, self.__df["Place"])
        return df_matrix


    def __create_tfidf_matrix(self):
        """ Generation of a TFIDF matrix for the tokens in the documents

        Returns:
            None

        """

        token_string_list = p.list_to_string(self.__df["grams"])
        self.__df["tokens_string"] = [x.replace(" ", "_") for x in token_string_list]
        df_matrix = p.use_tfidfVectorizer(self.__df["tokens_string"])
        df_matrix = p.dataframe_for_training(df_matrix, self.__df["Place"])
        return df_matrix


    def preprocess_data(self, input_dir = 'results_scrapping.csv', wordreduction = 'Lemmatization'):
        """ Reading and preprocessing of the data

        Args:
            input_dir: Path to the raw data
            wordreduction: Selection of the desired word reduction algorithm, vales: Lemmatization and Stemming

        Returns:
            None

        """

        self.__read_data(input_dir)
        self.__mapping_country_continent()

        if wordreduction == 'Lemmatization':
            self.__preprocess_raw_data_lemma()
        elif wordreduction == 'Stemming':
            self.__preprocess_raw_data_stemming()
        else:
            raise ValueError('No valid wordreduction selected.')

        self.__create_no_NE_tourist_attractions()


    def create_train_data(self, grams = '1-grams', train_matrix = 'True/False'):
        """ Transformation of the preprocessed data into a training data matrix

        Args:
            grams (String): Selected grams for training, values: 1-grams, 2-grams and mixed-grams
            train_matrix (String): Selected training matrix, values: TFIDF and True/False

        Returns:
            DataFrame -> DataFrame containing the training set

        """

        if grams == '1-grams':
            self.__create_one_grams()
        elif grams == '2-grams':
            self.__create_two_grams()
        elif grams == 'mixed-grams':
            self.__create_mixed_grams()
        else:
            raise ValueError('No valid grams selected.')

        if train_matrix == 'TFIDF':
            return self.__create_tfidf_matrix()
        elif train_matrix == 'True/False':
            return self.__create_true_false_matrix()
        else:
            raise ValueError('No valid train matrix selected.')


if __name__ == '__main__':
    prep_lemma = Preprocessing()
    prep_lemma.preprocess_data('results_scrapping.csv', 'Lemmatization')
