import json
import numpy as np
import sys

np.warnings.filterwarnings('ignore')

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Embedding, GRU
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import logparse as lp


class Derplog:

    def __init__(self,
                 lcs_file='scratch/lcs.json',
                 events_file='scratch/events.json',
                 weight_file=None,
                 epocs=50,
                 W=20,
                 summary=False):
        self.lcs_file = lcs_file
        self.events_file = events_file
        self.weight_file = weight_file
        self.W = W # log event window size
        self.epochs = epocs
        self.checkpoint_path = 'checkpoints/weights.{epoch:02d}-{val_acc:.3f}.hdf5'

        self.read_log_events()
        self.set_train_test_data()
        self.build_model(summary)

    def build_model(self, summary=False):
        self.model = Sequential()

        # self.model.add(Embedding(vocab_size, 128))
        # self.model.add(Embedding(vocab_size, 128, input_length=vocab_size-1))
        self.model.add(Embedding(self.vocab_size, 128, input_length=self.W-1))
        self.model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(self.vocab_size, activation='softmax'))

        if self.weight_file:
            self.model.load_weights(self.weight_file)

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if summary:
            self.model.summary()

        return self.model

    # Transform events into matrix of padded sequences of window size W,
    # then separate out y's and one-hot encode those.
    # def set_train_test_data(self, event_sequence, W, vocab_size, pct=0.8):
    def set_train_test_data(self, pct=0.8):
        sequences = [ self.event_sequence[max(0,i-self.W):i+1]
                      for i in range(len(self.event_sequence)-1) ]
        sequences = pad_sequences(sequences, maxlen=self.W)
        X = sequences[:,0:self.W-1]
        y = sequences[:,self.W-1:self.W]
        y = to_categorical(y, num_classes=self.vocab_size)

        split = round(len(X) * pct)
        self.X_train, self.X_test = X[:split], X[split:]
        self.y_train, self.y_test = y[:split], y[split:]

        return ((self.X_train, self.y_train), (self.X_test, self.y_test))


    def read_log_events(self):
        with open(self.events_file, 'r') as fd:
            events = np.array(json.load(fd))

        self.lcsmap = lp.LCSMap.from_file(self.lcs_file)
        self.event_sequence = events[:,0]
        self.vocab_size = len(self.lcsmap.lcsmap)

        return (self.lcsmap, self.event_sequence, self.vocab_size)


    # def fit_model(model, X_train, y_train, X_test, y_test):
    def fit(self):
        print('Fitting model')

        checkpoint = ModelCheckpoint(
            self.checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            period=1
        )

        self.model.fit(self.X_train, self.y_train,
                       callbacks=[checkpoint],
                       epochs=self.epochs,
                       validation_data=(self.X_test, self.y_test)
        )
        score, acc = self.model.evaluate(self.X_test, self.y_test)

        return score, acc


    def predict(self, seq):
        if not hasattr(seq, '__getitem__'):
            seq = np.array(seq)
        seq = pad_sequences(seq, maxlen=self.W-1)
        y_hat = self.model.predict_classes(seq)

        return y_hat


    def predict_proba(self, seq):
        if not hasattr(seq, '__getitem__'):
            seq = np.array(seq)
        seq = pad_sequences(seq, maxlen=self.W-1)
        y_hat_proba = self.model.predict_proba(seq)

        return y_hat_proba

    def predict_top_classes(self, seq, tol=0.1):
        proba = self.predict_proba([seq])[0]
        classes = reversed(sorted(enumerate(proba), key=lambda x: x[1]))

        return [ x[0] for x in classes if x[1] > tol ]


if __name__ == '__main__':
    if len(sys.argv) > 1:
        weight_file = sys.argv[1]
    else:
        weight_file = None

    derplog = Derplog(epocs=10, weight_file=weight_file)

    if not weight_file:
        score, acc = derplog.fit()
        print('Test score:', score)
        print('Test accuracy:', acc)

    # lcsmap, event_sequence, vocab_size = read_log_events()
    # print("Log keys: %d, log events: %d, training window: %d" %
    #       (vocab_size, len(event_sequence), W))

    # model = build_model(weight_file=weight_file)

    # if not weight_file:
    #     (X_train, y_train), (X_test, y_test) = train_test_data()
    #     score, acc = fit_model(model, X_train, y_train, X_test, y_test)

    #     print('Test score:', score)
    #     print('Test accuracy:', acc)
