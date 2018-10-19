
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


class InputRep:
    """A simple feature extractor and vectorizer"""

    def __init__(self):
        # self.stopset = sorted(set(stopwords.words('english')))
        self.lemmatizer = WordNetLemmatizer()
        self.max_features = 10000
        # create the vectorizer
        self.vectorizer = CountVectorizer(
            max_features= self.max_features,
            strip_accents=None,
            analyzer="word",
            tokenizer=self.mytokenize,
            stop_words=None,
            ngram_range=(1, 2),
            binary=False,
            lowercase=True,
            preprocessor=None
        )

    def mytokenize(self, text):
        """Customized tokenizer.
        Here you can add other linguistic processing and generate more normalized features
        """
        tokens = word_tokenize(text)
        tokens = [t.lower() for t in tokens]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return tokens

    def fit(self, train_texts, unlabeled=None):
        # fit to train corpus
        self.vectorizer.fit(train_texts)
        # print(self.vectorizer.get_feature_names()) # to manually check if the tokens are reasonable

    def get_vects(self, texts):
        '''
        Tokenizes and creates a BoW vector.
        :param texts: A list of strings each string representing a text.
        :return: X: A sparse csr matrix of TFIDF or Count -weighted ngram counts.
        '''
        X = self.vectorizer.transform(texts)
        return X.toarray()
