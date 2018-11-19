import argparse
import os
from collections import Counter
import pickle
import math
from collections import defaultdict
import random

class Trainer:

    def __init__(self, stopwords_file, train_dir, dev_dir, learn_rate, reg_factor):

        self.weight_vec = defaultdict(float)

        with open(stopwords_file) as f:
            stop_words = f.read().rstrip().split(",")
        self.ignore = set(".,:;()[]?!-'\!\\/*").union(set(stop_words))

        # stores paths of all training reviews, for example pos/cv276_15684.txt or neg/cv496_11185.txt
        # which helps us to shuffle across all pos and neg reviews, load each review only when needed and identify its class
        self.all_train_files = []
        for (train_path, train_subdir, train_files) in os.walk(train_dir):
            if train_path.endswith("pos"):
                for f in train_files:
                    self.all_train_files.append(os.sep.join(["pos", f]))
            else:
                for f in train_files:
                    self.all_train_files.append(os.sep.join(["neg", f]))

        self.train(train_dir, dev_dir, learn_rate, reg_factor)

    def extract_freqs(self, polarity, file):

        feature_vec = Counter()
        feature_vec.update([word + " " + polarity for line in file for word in line.strip().split() \
                                if word not in self.ignore])

        for feature, value in feature_vec.items():
         feature_vec[feature] = math.log(value + 1)

        return feature_vec

    def prepare_vecs(self, dir, file):

        with open(os.sep.join([dir, file])) as f:
            feature_vec_pos = self.extract_freqs("pos", f)
            f.seek(0) #sets reader in list comprehension in self.extract_freqs back to beginning of review
            feature_vec_neg = self.extract_freqs("neg", f)

        return feature_vec_pos, feature_vec_neg

    def dot_prod(self, feature_vec):

        dot_product = 0.0
        for feature, value in feature_vec.items():
            if feature in self.weight_vec:
                dot_product += value * self.weight_vec[feature]

        return dot_product*0.01 #shrink value to avoid "OverflowError: math range error"

    def class_probs(self, feature_vec_pos, feature_vec_neg):

        exp_pos = math.exp(self.dot_prod(feature_vec_pos))
        exp_neg = math.exp(self.dot_prod(feature_vec_neg))

        prob_pos = exp_pos/(exp_pos + exp_neg)
        prob_neg = exp_neg/(exp_pos + exp_neg)

        return prob_pos, prob_neg

    def expected_feature_vec(self, feature_vec_pos, feature_vec_neg):

        prob_pos, prob_neg = self.class_probs(feature_vec_pos, feature_vec_neg)
        expected_feature_vec = defaultdict()

        for feature_pos, value_pos in feature_vec_pos.items():
            expected_feature_vec[feature_pos] = prob_pos * value_pos

        for feature_neg, value_neg in feature_vec_neg.items():
            expected_feature_vec[feature_neg] = prob_neg * value_neg

        return expected_feature_vec


    def update_weights(self, expected_feature_vec, observed_feature_vec, learn_rate, reg_factor):

        for feature, value in observed_feature_vec.items():
            self.weight_vec[feature] += learn_rate*(value - expected_feature_vec[feature] - self.weight_vec[feature]*reg_factor)


    def classify_dev(self, dev_path, file):

        feature_vec_pos, feature_vec_neg = self.prepare_vecs(dev_path, file)

        score_pos = self.dot_prod(feature_vec_pos)
        score_neg = self.dot_prod(feature_vec_neg)

        return "positive" if score_pos > score_neg else "negative"


    def train(self, train_dir, dev_dir, learn_rate, reg_factor):

        accuracy = 0.0
        for i in range(20):
            correct = 0
            wrong = 0
            random.shuffle(self.all_train_files)
            for file in self.all_train_files:

                    feature_vec_pos, feature_vec_neg = self.prepare_vecs(train_dir, file)
                    expected_feature_vec = self.expected_feature_vec(feature_vec_pos, feature_vec_neg)

                    if file.split("/")[0] == "pos": #identify correct class by looking at path prefix as stored in self.all_train_files
                        self.update_weights(expected_feature_vec, feature_vec_pos, learn_rate, reg_factor)
                    else:
                        self.update_weights(expected_feature_vec, feature_vec_neg, learn_rate, reg_factor)

            for (dev_path, _, dev_files) in os.walk(dev_dir):
                for file in dev_files:
                    pred = self.classify_dev(dev_path, file)
                    if (dev_path.endswith("pos") and pred == "positive") \
                            or (dev_path.endswith("neg") and pred == "negative"):
                        correct += 1
                    else:
                        wrong += 1

            current_accuracy = correct / (correct + wrong)
            print("Epoch:", i, "Accuracy:", current_accuracy)
            if current_accuracy > accuracy:
                accuracy = current_accuracy
                print("#New Best Accuracy:", accuracy)
                with open(args.model, "wb") as out:
                    pickle.dump(self.weight_vec, out)


parser = argparse.ArgumentParser(
    description="Train and evaluate log-linear model for sentiment classification")
parser.add_argument(
    "train_dir", type=str, help="directory with positive and negative training data in sub-folders")
parser.add_argument(
    "dev_dir", type=str, help="directory with positive and negative dev data in sub-folders")
parser.add_argument("stopwords", type=str, help="file of words to ignore")
parser.add_argument("--learn_rate", type=float, help="learning rate for SGD")
parser.add_argument("--reg_factor", type=float, help="regularization factor for SGD")
parser.add_argument("model", type=str,
                    help="file to save the trained model")
args = parser.parse_args()

log_linear = Trainer(args.stopwords, args.train_dir, args.dev_dir, args.learn_rate, args.reg_factor)
