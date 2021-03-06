from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.layers import Lambda
from keras.optimizers import Adadelta
from keras.models import Model

def recall_metric(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_metric(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_metric(y_true, y_pred):
    p = precision_metric(y_true, y_pred)
    r = recall_metric(y_true, y_pred)
    return 2*((p*r)/(p+r+K.epsilon()))

def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

def siamese_model(word_embedding_matrix,
                nwords: int, max_sentence_length: int,
                embedding_dim: int,
                number_lstm_units: int,
                number_dense_units: int, rate_drop_dense: int
               ):
    """
    Build Siamese model

    Return:
        model
    """

    from keras.layers.embeddings import Embedding
    from keras.layers import Input, Dense, Dropout, LSTM, Bidirectional

    # create word embedding layer
    embedding_layer = Embedding(input_dim=nwords, output_dim=embedding_dim,
                                weights=[word_embedding_matrix],
                                input_length=max_sentence_length, trainable=False)

    # create LSTM encoder
    lstm_layer = Bidirectional(LSTM(number_lstm_units))

    # create LSTM encoder layer for sentence one
    s1_input = Input(shape=(max_sentence_length,), dtype='int32')
    embedded_s1 = embedding_layer(s1_input)
    x1 = lstm_layer(embedded_s1)

    # create LSTM encoder layer for second two
    s2_input = Input(shape=(max_sentence_length,), dtype='int32')
    embedded_s2 = embedding_layer(s2_input)
    x2 = lstm_layer(embedded_s2)

    # # model 1:
    # # merge two LSTM encodes vectors from sentences
    # merged = concatenate([x1, x2])
    # merged = Dropout(rate_drop_dense)(merged)
    # merged = Dense(number_dense_units, activation='relu')(merged)
    #
    # merged = Dropout(rate_drop_dense)(merged)
    # preds = Dense(1, activation='sigmoid')(merged)
    #
    # model = Model(inputs=[s1_input, s2_input], outputs=preds)
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc',f1, precision, recall])

    ## model 2:
    # calculate the distance as defined by the MaLSTM model
    # ref: https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07
    malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                             output_shape=lambda x: (x[0][0], 1))([x1, x2])

    # pack it all up into a model
    model = Model([s1_input, s2_input], [malstm_distance])

    optimizer = Adadelta()
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy', f1_metric])

    return model
