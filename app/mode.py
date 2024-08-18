import re
from bs4 import BeautifulSoup
import pickle
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import numpy as np
from nltk.stem.porter import PorterStemmer
import nltk

# Download required datasets
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


with open('app/tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

with open('app/word2vec_model.pkl', 'rb') as file:
    word2vec_model = pickle.load(file)


def common_words(q1, q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return len(w1 & w2)


def total_words(q1, q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return len(w1) + len(w2)


# features based on tokens
def token_features(q1, q2):

    safe_div = 0.0001

    token_features = [0.0]*8

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    stopword = stopwords.words('english')

    q1_non_stopwords = set([word for word in q1_tokens if word not in stopword])
    q2_non_stopwords = set([word for word in q2_tokens if word not in stopword])

    q1_stop_words = set([word for word in q1_tokens if word in stopword]) 
    q2_stop_words = set([word for word in q2_tokens if word in stopword]) 

    common_word_count = len(q1_non_stopwords.intersection(q2_non_stopwords))
    common_stop_word_count = len(q1_stop_words.intersection(q2_stop_words))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count/(min(len(q1_non_stopwords), len(q2_non_stopwords)) + safe_div)
    token_features[1] = common_word_count/(max(len(q1_non_stopwords), len(q2_non_stopwords)) + safe_div)
    token_features[2] = common_stop_word_count/(min(len(q1_stop_words), len(q2_stop_words)) + safe_div)
    token_features[3] = common_stop_word_count/(max(len(q1_stop_words), len(q2_stop_words)) + safe_div)
    token_features[4] = common_token_count/(min(len(q1_tokens), len(q2_tokens)) + safe_div)
    token_features[5] = common_token_count/(max(len(q1_tokens), len(q2_tokens)) + safe_div)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features


# Fuzzy Features
def fuzzy_features(q1, q2):
    
    fuzzy_features = [0.0]*4
    
    # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    # token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    # token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features

# applying stemming
ps = PorterStemmer()

def stem_words(text):
    return " ".join([ps.stem(word) for word in text.split()])


# data preprocessing
def preprocess(q):
    
    q = str(q).lower().strip()

    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    q = q.replace('?', '')

    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')

    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

     # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

        # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()

    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    return q

def tfidf_weighted_word2vec(doc, word2vec_model, tfidf_scores):
    word_vectors = []
    for word in doc:
        if word in word2vec_model.wv.key_to_index:
            # Get the Word2Vec vector for the word
            vector = word2vec_model.wv[word]
            # Multiply by the TF-IDF score for the word
            tfidf_score = tfidf_scores.get(word, 0.0)
            weighted_vector = vector * tfidf_score
            word_vectors.append(weighted_vector)
    
    # Compute the weighted average of the word vectors
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)


def preprocessing(q1, q2):

    features = []

    q1 = preprocess(q1)
    q2 = preprocess(q2)

    q1 = stem_words(q1)
    q2 = stem_words(q2)

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    features.append(len(q1))
    features.append(len(q2))

    features.append(len(q1.split(" ")))
    features.append(len(q2.split(" ")))

    features.append(common_words(q1, q2))
    features.append(total_words(q1, q2))
    features.append(common_words(q1, q2)/(total_words(q1, q2) + 0.0001))

    features.extend(token_features(q1, q2))
    features.extend(fuzzy_features(q1, q2))

    tfidf_matrix = tfidf_vectorizer.transform([q1, q2])
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

    tfidf_scores = dict(zip(tfidf_feature_names, tfidf_matrix.toarray().sum(axis=0)))

    q1_vec = tfidf_weighted_word2vec(q1_tokens, word2vec_model, tfidf_scores)
    q2_vec = tfidf_weighted_word2vec(q2_tokens, word2vec_model, tfidf_scores)

    return np.hstack((np.array(features).reshape(1, 19), (q1_vec).reshape(1, -1), (q2_vec).reshape(1, -1)))
