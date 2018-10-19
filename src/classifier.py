import pickle

import numpy as np
np.random.seed(15)

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import optimizers
from sklearn.preprocessing import LabelBinarizer
from inputrep import InputRep

class Classifier:
    """The Classifier"""

    def __init__(self):
        self.labelset = None
        self.inputrep = InputRep()
        self.label_binarizer = LabelBinarizer()
        self.model = None
        self.epochs = 2
        self.batchsize = 32

        self.dense=50
        # self.dense1=40
        # self.layer='relu'
        self.layer='hard_sigmoid'

    def create_model(self):
        """Create a neural network model and return it.
        Here you can modify the architecture of the model (network type, number of layers, number of neurones)
        and its parameters"""

        # Define input vector, its size = number of features of the input representation
        input = Input((self.inputrep.max_features,))
        # Define output: its size is the number of distinct (class) labels (class probabilities from the softmax)
        layer1 = Dense(self.dense, activation=self.layer)(input)
        output = Dense(len(self.labelset), activation='softmax')(layer1)
        # create model by defining the input and output layers
        model = Model(inputs=input, outputs=output)
        # compile model (pre
        model.compile(optimizer=optimizers.Adamax(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, texts, labels):
        """Train the model using the list of text examples together with their true (correct) labels"""

        # create the binary output vectors from the correct labels
        Y_train = self.label_binarizer.fit_transform(labels)
        # get the set of labels
        self.labelset = set(self.label_binarizer.classes_)
        print("LABELS: %s" % self.labelset)
        # build the feature index (unigram of words, bi-grams etc.)  using the training data
        self.inputrep.fit(texts)
        # create a model to train
        self.model = self.create_model()
        # for each text example, build its vector representation
        X_train = self.inputrep.get_vects(texts)
        # Train the model!
        self.model.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batchsize)

    def predict(self, texts):
        """Use this classifier model to rpedict class labels for a list of input texts.
        Returns the list of predicted labels
        """

        # first, get the vector representations of the input texts, using the same inputrep object as the
        # one created with the training data
        X = self.inputrep.get_vects(texts)
        # get the predicted output vectors: each vector will contain a probability for each class label
        Y = self.model.predict(X)
        # from the output probability vectors, get the labels that got the best probability scores
        return self.label_binarizer.inverse_transform(Y)
