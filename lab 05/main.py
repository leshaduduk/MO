import re
import sys
import os
from xml import etree
from itertools import islice
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from nltk.stem import PorterStemmer


def gaussian(sigma_squared):
    def kernel(x, y):
        distances = euclidean_distances(x, y)
        return np.exp(-distances ** 2 / 2 / sigma_squared)

    return kernel


def get_words(letter):
    letter = letter.lower()
    letter = re.sub(r"<[^>]+>", "", letter)
    letter = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "httpaddr",
        letter,
    )
    letter = re.sub(r"[^@\s]+@[^@\s]+\.[^@\s]+", "emailaddr", letter)
    letter = re.sub(r"\d+", " number ", letter)
    letter = re.sub(r"\$", " dollar ", letter)
    letter = re.sub(r"[^a-zA-Z]", " ", letter)
    ps = PorterStemmer()
    return [ps.stem(word) for word in letter.split()]


def generate_words():
    mail_dir = sys.argv[1]
    spam_dir = sys.argv[2]
    stems = defaultdict(lambda: 0)
    ps = PorterStemmer()
    for d in [mail_dir, spam_dir]:
        for name in os.listdir(d)[:500]:
            print(name)
            path = os.path.join(d, name)
            with open(path, "r", encoding="cp1250") as f:
                try:
                    text = f.read()
                    for word in get_words(text):
                        stems[ps.stem(word)] += 1
                except:
                    print("pass on", path)
    with open(sys.argv[3], "w+") as f:
        for stem, _ in islice(
            sorted(stems.items(), key=lambda x: x[1], reverse=True), 2000
        ):
            f.write(stem + "\n")


def write_features():
    mail_dir = sys.argv[1]
    spam_dir = sys.argv[2]
    our_file = sys.argv[3]
    rows = []
    results = []
    vocab_df = pd.read_csv("custom.txt")
    vocab = {word: index for index, word in vocab_df.itertuples()}
    ps = PorterStemmer()
    for name in os.listdir(mail_dir)[:500]:
        print(name)
        path = os.path.join(mail_dir, name)
        with open(path, "r", encoding="cp1250") as f:
            text = None
            try:
                text = f.read()
                features = np.zeros(2000)
                for word in get_words(text):
                    stem = ps.stem(word)
                    if stem in vocab:
                        features[vocab[stem]] = 1
                rows.append(features)
                results.append(0)
            except:
                print("pass on", path)
    for name in os.listdir(spam_dir)[:500]:
        print(name)
        path = os.path.join(spam_dir, name)
        with open(path, "r", encoding="cp1250") as f:
            text = None
            try:
                text = f.read()
                features = np.zeros(2000)
                for word in get_words(text):
                    stem = ps.stem(word)
                    if stem in vocab:
                        features[vocab[stem]] = 1
                rows.append(features)
                results.append(1)
            except:
                print("pass on", path)
    df = pd.DataFrame(rows)
    df.insert(2000, "result", results)
    df.to_csv("train.csv")


def main():
    # generate_words()
    write_features()


if __name__ == "__main__":
    main()
