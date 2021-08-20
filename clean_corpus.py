import re
import pickle
import heapq
import itertools
import glob
import unicodedata
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


def load_keyword_sentences(file_specification):
    if file_specification == "MNI_sentences_includesWordBoundaries":
        file_list = glob.glob(
            "sentence_data/MNI_sentences_Neuroimage*_includesWordBoundaries.p"
        )
        keyword_sentences = []
        for file_name in file_list:
            print("*** loading: " + file_name)
            keyword_sentences.extend(pickle.load(open(file_name, "rb")))
    return keyword_sentences


def flatten_list(unflat_list):
    flat_list = list(itertools.chain.from_iterable(unflat_list))
    return flat_list


def flatten_list_keepDocRelation(keyword_sentences):
    for id_sub, sub in enumerate(keyword_sentences):
        if isinstance(sub, list):
            keyword_sentences[id_sub] = " ".join(sub)
    return keyword_sentences


def remove_punctuation(corpus):
    for i in range(len(corpus)):
        corpus[i] = re.sub(r"\W", " ", corpus[i])
        corpus[i] = re.sub(r"\s+", " ", corpus[i])
        corpus[i] = re.sub(r"[\W_]", " ", corpus[i])  # remove underscore
    return corpus


def remove_numbers(corpus):
    for i in range(len(corpus)):
        corpus[i] = re.sub(r"\d+", "", corpus[i])
    return corpus


def remove_singleLetters(corpus):
    for i in range(len(corpus)):
        corpus[i] = re.sub(r"\b[a-zA-Z]\b", " ", corpus[i])
    return corpus


def get_wordfreq(corpus):
    wordfreq = {}
    for sentence in corpus:
        tokens = word_tokenize(sentence)
        for token in tokens:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1
    return wordfreq


def get_mostfreq(wordfreq, N):
    most_freq = heapq.nlargest(N, wordfreq, key=wordfreq.get)
    return most_freq


def get_above_n_occurances(wordfreq, N):
    words_above_n_occurances = []
    for word, freq in wordfreq.items():
        if freq >= N:
            words_above_n_occurances.append(word)
    sort_words_above_n_occurances = sorted(
        words_above_n_occurances
    )  # sort list alphabetically
    return sort_words_above_n_occurances


def filter_stopwords(words, stop_words):
    clean_words = [w for w in words if not w in stop_words]
    return clean_words


def word_stemming(corpus, stemmer="porter"):
    stemmed_corpus = []

    if stemmer == "porter":
        print("*** stemming with porter")
        stemprocedure = PorterStemmer()
    elif stemmer == "lancaster":
        print("*** stemming with lancaster")
        stemprocedure = LancasterStemmer()
    elif stemmer == "wordnet":
        print("*** stemming with wordnet")
        stemprocedure = WordNetLemmatizer()

    for sentence in corpus:
        stemmed_tokens = []
        tokens = word_tokenize(sentence)
        for id_token, token in enumerate(tokens):
            if stemmer == "wordnet":
                stemmed_tokens.append(stemprocedure.lemmatize(token))
            else:
                stemmed_tokens.append(stemprocedure.stem(token))
        join_stemmed_tokens = " ".join(stemmed_tokens)  # join back to sentence
        stemmed_corpus.append(join_stemmed_tokens)
    return stemmed_corpus


def strip_accents(corpus):
    clean_corpus = []
    for text in corpus:
        try:
            text = unicode(text, "utf-8")
        except NameError:  # unicode is a default on python 3
            pass

        text = (
            unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
        )
        clean_corpus.append(str(text))
    return clean_corpus
