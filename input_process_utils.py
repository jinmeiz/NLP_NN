from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import numpy as np


# tokenize
def tokenize_doc(doc_words: list):
    """
    Tokenize documents in list of words

    Return:
        tokenizer
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(doc_words)

    print('document_count:', tokenizer.document_count)
    print('number of words:', len(tokenizer.word_index))

    return tokenizer


def obtain_word_embedding(tokenizer, doc_words: list, embedding_dim: int):
    """
    Get word cofficients from Word2Vec

    Return:
        matrix (nwords+1, embedding_dim)
    """

    # word to vector
    model = Word2Vec(doc_words, min_count=1, size=embedding_dim)
    Word_Vec = model.wv

    # intialize embeddding matrix
    nwords = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((nwords, embedding_dim))
    print("Embedding matrix shape: %s" % str(embedding_matrix.shape))

    # matching word to vector
    for word, i in tokenizer.word_index.items():
        try:
            embedding_vector = Word_Vec[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        except KeyError:
            print("vector not found for word - %s" % word)

        # print sample for spot check
        if i == 5:
            print('sample word embeddding:')
            print(word, i)
            print(embedding_vector)

    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    return embedding_matrix


def input_process(tokenizer, sentences1: list, sentences2: list, max_sentence_length: int):
    """
    Convert sentence to tokens, and do padding

    Arguments:
        tokenizer
        sentences1: list of sentence string
        sentences2: list of sentence string
        max_sentence_length
    """
    s1 = tokenizer.texts_to_sequences(sentences1)
    s2 = tokenizer.texts_to_sequences(sentences2)

    padded_s1 = pad_sequences(s1, maxlen=max_sentence_length)
    padded_s2 = pad_sequences(s2, maxlen=max_sentence_length)

    return padded_s1, padded_s2

def create_train_valid_test_set(s1_padded_tokens, s2_padded_tokens, labels,
                                test_size = 0.5, ):
    """
    Split data into train, valid, and test sets
    """
    train_pair = [(x1, x2) for x1, x2 in zip(s1_padded_tokens, s2_padded_tokens)]

    # split data for train, validation, and test
    x_train, x_valid_test, y_train, y_valid_test = train_test_split(train_pair, labels,
                                                                    test_size=test_size, random_state=42)

    x_valid, x_test, y_valid, y_test = train_test_split(x_valid_test, y_valid_test,
                                                        test_size=0.5,
                                                        random_state=42)

    # convert to numpy array
    x1_train = np.array([x[0] for x in x_train])
    x2_train = np.array([x[1] for x in x_train])
    y_train = np.array(y_train)

    x1_valid = np.array([x[0] for x in x_valid])
    x2_valid = np.array([x[1] for x in x_valid])
    y_valid = np.array(y_valid)

    x1_test = np.array([x[0] for x in x_test])
    x2_test = np.array([x[1] for x in x_test])
    y_test = np.array(y_test)

    return x1_train, x2_train, y_train, x1_valid, x2_valid, y_valid, x1_test, x2_test, y_test
