import pickle
from sklearn.metrics import accuracy_score
import dataloader
from classifier import Classifier
import time

import numpy as np
np.random.seed(15)


def save_classifier(classifier, classifier_file):
    """Save a classifier model to binary file on disk"""
    pickle.dump(classifier, open(classifier_file, "wb"))

def load_classifier(classifier_file):
    """Load a classifier model from disk"""
    return pickle.load(open(classifier_file, "rb"))

def check_train(trainfile, classifier_file):
    """Create an instance of 'Classifier', train it with the dataset in 'trainfile'
    and save the trained model to 'classifier_file" on disk."""
    # Load training data
    texts, labels = dataloader.load(trainfile)
    # Create a Classifier instance and train it on data
    classifier = Classifier()
    classifier.train(texts, labels)
    # Save classifier
    save_classifier(classifier, classifier_file)
    print("Done.\n--------------------------------------\n")


def eval(classifier_file, testfile):
    """Loads a classifier model from disk, and evaluate it with test data from
    'testfile' file"""
    try:
        # load classifier
        classifier = load_classifier(classifier_file)
        # load test data
        texts, labels = dataloader.load(testfile)
        # classify input texts with the classifier (let the classifier predict classes for the input texts)
        predictions = classifier.predict(texts)
        # compute accuracy score by comparing the model predictions with the true labels
        acc = accuracy_score(labels, predictions)
        print("\nAccuracy: %.4f\n" % acc)
    except  FileNotFoundError:
        print(" Error: cannot find model file '%s'" % classifier_file)
    print("Done.\n--------------------------------------\n")



def check_predict(classifier_file, texts):
    # load classifier
    try:
        classifier = load_classifier(classifier_file)
        preds = classifier.predict(texts)
        for text, pred in zip(texts, preds):
            print(" %s:\t%s" % (pred, text))
    except FileNotFoundError:
        print(" Error: cannot find model file '%s'" % classifier_file)
    print("\nDone.\n--------------------------------------\n")



text_examples = ["The worst restaurant in Grenoble.", "Food was good but a bit expensive.", "The food was excellent."]

if __name__ == "__main__":
    datadir = "./data/"
    resourcedir = "./"
    classifier_file = resourcedir + "classifier.bin"
    trainfile =  datadir + "reviews_train20K.csv"
    # testfile = datadir + "test5K.csv"
    testfile =  datadir + "reviews_dev3K.csv"
    # Basic checking
    time.process_time()
    print("\n")
    print("1. Classifying 3 examples with the model provided by student...\n")
    check_predict(classifier_file, text_examples)
    # Evaluation of the classifier provided by student
    print("2. Evaluating the model provided by student...\n")
    eval(classifier_file, testfile)
    # Check training
    print("3. Checking training...\n")
    check_train(trainfile, classifier_file)
    # Chack predictions with trained model
    print("4. Classifying 3 examples with the trained model...\n")
    check_predict(classifier_file, text_examples)
    # Evaluation of the classifier after training
    print("5. Evaluating the trained classifier...\n")
    eval(classifier_file, testfile)
    print("\nExec time: %.2f s." % time.process_time())




