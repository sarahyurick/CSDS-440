from os.path import dirname, abspath, join
from typing import List
from sklearn.feature_extraction import DictVectorizer
import itertools
from itertools import islice


# root_dir = dirname(abspath(f"{__file__}/."))
# data_folder = join(dirname(root_dir), "data-sets")

# Folders
# task_a_lab_folder = join(data_folder, "task_a_lab")

# ---------------------------------------------- #
#                   Spam Data                    #
# ---------------------------------------------- #


# domains = "task_a_labeled_train.tf" , "task_a_u00_eval_lab.tf"
# example of how to use:
# vectors, labels = data.collect_spam_a_data("task_a_labeled_train.tf")


def collect_spam_a_data(domain, num_features=3000):
    """[summary]

    Args:
        domain: file name. "task_a_labeled_train.tf", "task_a_u00_eval_lab.tf"
        num_features (int, optional): restricts number of features. Defaults to 3000.

    Returns:
        tuple: list of vectors (each vector is one example where index represents
        a word and value is the frequency), list of labels corresponding to each vector
    """
    vocab = {}
    filename = domain  # join(task_a_lab_folder, domain)
    spams, labels = read_spams(domain, vocab, filename)
    v = DictVectorizer(sparse=False)
    vectors = v.fit_transform(spams)
    vectors = [v[:num_features] for v in vectors]
    return vectors, labels


def read_spams(domain, vocab, filename):
    spams = []
    labels = []
    with open(filename, "r") as f:
        for line in f:
            words_dict, bool_label = parse_spam(line, vocab)
            spams.append(words_dict)
            labels.append(bool_label)
    return spams, labels


def parse_spam(line, vocab):
    words_dict = {}
    split = line.split(" ")
    for word_count in split[1:]:
        word, count_str = word_count.split(":")
        count = int(count_str)
        vocab[word] = count if word not in vocab else vocab[word] + count
        words_dict[word] = count
    label = int(split[0])
    bool_label = 1 if label == 1 else 0
    return words_dict, bool_label


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def collect_amazon_data(filename1, filename2, num_features=3000, max_num_reviews=1000):
    """
    example use:

    toys_reviews_vectors, toys_reviews_labels, patio_reviews_vectors, patio_reviews_labels
    = collect_amazon_data('Toys_and_Games_5.json', 'Patio_Lawn_and_Garden_5_1.json')

    """
    reviews1, labels1, temp_counts = read_amazon(filename1)
    reviews2, labels2, counts = read_amazon(filename2, existing_dict=temp_counts)

    reviews1 = reviews1[:max_num_reviews]
    reviews2 = reviews2[:max_num_reviews]
    labels1 = labels1[:max_num_reviews]
    labels2 = labels2[:max_num_reviews]

    counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)}
    words = list(counts.keys())[:num_features]

    reviews1 = parse_amazon(reviews1, words, num_features=num_features)
    reviews2 = parse_amazon(reviews2, words, num_features=num_features)

    return reviews1, labels1, reviews2, labels2


def read_amazon(filename, existing_dict=None):
    import re
    import json

    regex = re.compile('([^\s\w]|_)+')

    reviews = []
    labels = []
    if existing_dict is None:
        counts = dict()
    else:
        counts = existing_dict

    my_file = open(filename, "r")
    for aline in my_file:
        data = json.loads(aline)
        review = regex.sub('', data["reviewText"]).lower()
        reviews.append(review)
        review = review.split()
        for word in review:
            try:
                current_count = counts[word]
                counts.update({word: current_count + 1})
            except KeyError:
                counts.update({word: 1})
        rating = data["overall"]
        if rating > 2.5:
            labels.append(1)
        else:
            labels.append(0)
    my_file.close()

    return reviews, labels, counts


def parse_amazon(reviews, words, num_features):
    all_reviews = []
    for review in reviews:
        words_in_review = [0] * num_features

        for word in review.split():
            try:
                word_index = words.index(word)
                temp_count = words_in_review[word_index]
                words_in_review[word_index] = temp_count + 1
            except ValueError:
                pass

        all_reviews.append(words_in_review)

    return all_reviews
