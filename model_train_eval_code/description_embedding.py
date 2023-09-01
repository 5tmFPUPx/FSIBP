# -*- coding: utf-8 -*-
# Embedding and clustering the field descriptions of wireshark.
# The clustering results are output to a csv file, and a field description corresponds to a label (a number with a label of 0-k).

import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
#import matplotlib.pyplot as plt
from kneed import KneeLocator
import numpy as np
import os

def generate_description_label(description_list, label_list): 
    data = list(zip(description_list, label_list))
    
    with open(os.path.join('..', 'data', 'description-label', 'description_label.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)


def descritpion_cossim(train_description_list, test_description_list, sentence_embeddin_model_name):
    current_directory = os.path.abspath(os.getcwd())
    parent_directory = os.path.dirname(current_directory)
    if os.path.exists(os.path.join(parent_directory, sentence_embeddin_model_name)) and os.path.isdir(os.path.join(parent_directory, sentence_embeddin_model_name)):
        model = SentenceTransformer(os.path.join('..', sentence_embeddin_model_name))
    else:
        model = SentenceTransformer(sentence_embeddin_model_name)
    num_train_description = len(train_description_list)
    train_description_embeddings = model.encode(train_description_list)
    train_cosine_similarities = cosine_similarity(train_description_embeddings)

    cost =[]
    if num_train_description//3 < 40:
        max_k = num_train_description//3
    else:
        max_k = 40
    for i in range(2, max_k+1):
        KM = KMeans(n_clusters = i, max_iter = 300)
        KM.fit(train_cosine_similarities)
        cost.append(KM.inertia_)    # inertia_: Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.
    # plot the cost against K values
    kneedle = KneeLocator(range(2, max_k+1), cost, curve='convex', direction='decreasing')
    optimal_k = kneedle.elbow
    if optimal_k == None:
        optimal_k = 4
    print('optimal_k:',optimal_k)

    kmeans = KMeans(optimal_k, random_state=0).fit(train_cosine_similarities)
    train_labels = kmeans.predict(train_cosine_similarities)

    all_description_list = train_description_list + test_description_list
    all_description_embeddings = model.encode(all_description_list)
    all_cosine_similarities = cosine_similarity(all_description_embeddings)

    num_all_description = len(all_cosine_similarities)
    test_labels = []
    for i in range(num_train_description, num_all_description):
        max_cos_value = max(all_cosine_similarities[i][:num_train_description])
        nearest_description_index = np.where(all_cosine_similarities[i][:num_train_description]==max_cos_value)[0].tolist()
        test_labels.append(train_labels[nearest_description_index[0]])

    description_list = train_description_list + test_description_list
    labels = list(train_labels) + test_labels

    generate_description_label(description_list, labels)

    return optimal_k
