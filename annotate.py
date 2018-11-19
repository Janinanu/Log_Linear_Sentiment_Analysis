import os
import argparse
import pickle
from collections import Counter
import math

class Classificator:

    def __init__(self, weights, test_dir):

        with open(weights, "rb") as model:
            self.weights = pickle.load(model)

        self.classify(test_dir)

    def extract_freqs(self, polarity, file):

        feature_vec = Counter()
        feature_vec.update([word + " " + polarity for line in file for word in line.strip().split() \
                                if word + " " + polarity in self.weights])

        for feature, value in feature_vec.items():
            feature_vec[feature] = math.log(value + 1)

        return feature_vec


    def prepare_vecs(self, test_path, file):

        with open(os.sep.join([test_path, file])) as f:

            feature_vec_pos = self.extract_freqs("pos", f)
            f.seek(0) #sets reader in list comprehension in self.extract_freqs back to beginning of review
            feature_vec_neg = self.extract_freqs("neg", f)

        return feature_vec_pos, feature_vec_neg

    def dot_prod(self, feature_vec):

        dot_product = 0.0
        for feature, value in feature_vec.items():
            if feature in self.weights:
                dot_product += value * self.weights[feature]

        return dot_product*0.01 #shrink value to avoid "OverflowError: math range error"

    def classify_review(self, test_path, file):

        feature_vec_pos, feature_vec_neg = self.prepare_vecs(test_path, file)

        score_pos = self.dot_prod(feature_vec_pos)
        score_neg = self.dot_prod(feature_vec_neg)

        return "positive" if score_pos > score_neg else "negative"

    def classify(self, test_dir):

        with open("classifications.txt", "w") as classifications:

            for (test_path, _, test_files) in os.walk(test_dir):
                for file in test_files:
                    pred = self.classify_review(test_path, file)
                    classifications.write(pred + "\n")

parser = argparse.ArgumentParser(
    description="Annotate reviews for sentiment classification")
parser.add_argument("model", type=str,
                    help="file to load the trained model")
parser.add_argument("test_dir", type=str, help="directory with positive and negative test data in sub-folders")
args = parser.parse_args()

classificator = Classificator(args.model, args.test_dir)
