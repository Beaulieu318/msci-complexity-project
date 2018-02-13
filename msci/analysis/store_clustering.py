from msci.analysis.networks import *
from msci.analysis.complexity import *
from msci.cleaning.store_ids import *
from msci.utils import log_bin

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import *
from sklearn.cluster import *
import networkx as nx
import community as comm
import itertools
import scipy as sp
import os
import operator
import re
import math

import wikipediaapi
import wikipedia

from rake_nltk import Rake
from msci.analysis import rake

import html2text


dir_path = os.path.dirname(os.path.realpath(__file__))

def wikipedia_result(search_term, lib_type='wikipedia'):
    if lib_type == 'wikipedia-api':
        wiki_wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI
            )

        page_py = wiki_wiki.page(search_term)
        if page_py.exists:
            print(page_py.summary[:20])
            return page_py.text
        else:
            return 0
    else:
        search = wikipedia.search(search_term)
        return search, wikipedia.page(search_term)


def wiki_search_exist(store_directory_list):
    exist_boolean = [wikipedia_result(i, lib_type='wikipedia-api') for i in store_directory_list]
    return exist_boolean


def keyword_processing(text_file, lib_type='rake', word_length_min=3, phrase_length_max=2, keyword_appearance_min = 2):
    if lib_type == 'rake':
        rake_object = rake.Rake(
            dir_path + '/../data/SmartStoplist.txt', word_length_min, phrase_length_max, keyword_appearance_min
            )
        t = re.sub(r'\b\w{15,100}\b', '', text_file.lower().replace('/',''))
        t = clean_text(t)
        keywords = rake_object.run(t)
        keywords = clean_keywords(keywords)
        return keywords
    else:
        r = Rake()
        r.extract_keywords_from_text(text_file)
        keywords = r.get_ranked_phrases()
        ranks = r.get_ranked_phrases_with_scores()
        return keywords, ranks


def html_txt(html_file):
    #html = function_to_get_some_html()
    html = open(dir_path + '/../data/htmls/' + html_file)
    text = html2text.html2text(html)
    return text
    

def keyword_dictionary(store_directory_list='mall_of_mauritius_directory.csv', search_data='search_result_data.csv', website_data='mall_of_mauritius_store_descriptions.csv'):
    directory_df = pd.read_csv(dir_path + '/../data/' + store_directory_list)
    search_df = pd.read_csv(dir_path + '/../data/' + search_data, encoding = "ISO-8859-1")
    website_df = pd.read_csv(dir_path + '/../data/' + website_data, encoding = "ISO-8859-1")
    store_list = directory_df.store_name.tolist()
    store_categories = directory_df.store_category.tolist()
    keyword_dictionary = {i: {'category': j} for (i, j) in list(zip(store_list, store_categories))}
    other = [[],[]]
    #return website_df, search_df, directory_df
    for i in range(len(website_df)):
        store = website_df.iloc[i].store_name
        if store in store_list:
            keyword_dictionary[store]['site keywords'] = keyword_processing(
                website_df.iloc[i].description, word_length_min=3, phrase_length_max=2, keyword_appearance_min = 1
                )
        else:
            other[0].append(store)
            print(store)
    for i in range(len(search_df)):
        store = search_df.iloc[i].store_name
        if store in store_list:
            keyword_dictionary[store]['search keywords'] = keyword_processing(search_df.iloc[i].keywords)
        else:
            other[1].append(store)
    return keyword_dictionary, other, store_list
    website_keywords = [keyword_processing(i) for i in website_df.description.tolist()]
    store_keywords = [keyword_processing(i) for i in search_df.keywords.tolist()]
    return website_keywords, store_keywords, store_list, 


def clean_text(text_file):
    remove_words_containing = ['#', '@', 'ä', 'ó', '»', '=', '&', '_', '+', '%', '©', 'ç', 'é', 'è', 'å', 'î', '¾', 'î', 'à', '±']
    for char in remove_words_containing:
        text_file = ' '.join(s for s in text_file.split() if not any(c is char for c in s))
    return text_file


def clean_keywords(keywords):
    split_list = [[[j, i[1]] for j in i[0].split()] for i in keywords]
    flat_split_list = [word for words in split_list for word in words]
    return flat_split_list


def combine_keywords(keyword_dictionary):
    if 'site keywords' in list(keyword_dictionary.keys()):
        keywords = keyword_dictionary['search keywords'] + keyword_dictionary['site keywords']
        seen = set()
        non_duplicate_keywords = [kw for kw in keywords if kw[0] not in seen and not seen.add(kw[0])]
        keyword_dictionary.pop('search keywords', None)
        keyword_dictionary.pop('site keywords', None)
        keyword_dictionary['keywords'] = non_duplicate_keywords
    else:
        keyword_dictionary['keywords'] = keyword_dictionary.pop('search keywords')
    return keyword_dictionary


def clean_keyword_dictionary(all_keyword_dictionary):
    stores = list(all_keyword_dictionary.keys())
    clean_dictionary = {}
    for i in stores:
        clean_dictionary[i] = combine_keywords(all_keyword_dictionary[i])
    return clean_dictionary


def keyword_network(formatted_keyword_dictionary):
    stores = list(formatted_keyword_dictionary.keys())
    all_keywords = [[j[0] for j in formatted_keyword_dictionary[i]['keywords']] for i in stores]
    flat_keywords = [word for words in all_keywords for word in words]
    keyword_list = list(set(flat_keywords))
    category_list = list(set([formatted_keyword_dictionary[i]['category'] for i in stores]))
    category_list = [i for i in category_list if type(i) is str]
    print(len(keyword_list) + len(category_list) + len(stores))

    #category_nodes = [(i, dict(node_type = 'category')) for i in category_list]
    #keyword_nodes = [(i, dict(node_type = 'keyword')) for i in keyword_list]
    #store_nodes = [(i, dict(node_type = 'store')) for i in stores]
    category_nodes = [(i, dict(node_type = 'category', label= i)) for i in category_list]
    keyword_nodes = [(i, dict(node_type = 'keyword', label = i)) for i in keyword_list]
    store_nodes = [(i, dict(node_type = 'store', label = i)) for i in stores]
    
    KN = nx.DiGraph()
    KN.add_nodes_from(category_nodes)
    KN.add_nodes_from(store_nodes)
    KN.add_nodes_from(keyword_nodes)

    category_edges = [(store, formatted_keyword_dictionary[store]['category'], 5) for store in stores]
    keyword_edges = [(store, edge[0], int(edge[1])) for store in stores for edge in formatted_keyword_dictionary[store]['keywords']]

    KN.add_weighted_edges_from(category_edges)
    KN.add_weighted_edges_from(keyword_edges)

    #nodes = list(KN.nodes)
    #node_name_dict = {i: nodes[i] for i in range(len(nodes))}

    return KN, stores, category_list, keyword_list


def graph_to_GraphML(G, name='shop_network.graphml'):
    nx.write_graphml(G, name)


def word_network_degree_distribution(G, store_list, category_list, keyword_list, plot=True, degree_type='keyword'):
    degrees = list(nx.degree(G))
    keyword_degrees = [i for i in degrees if i[0] in keyword_list]
    store_degrees = [i for i in degrees if i[0] in store_list]
    category_degrees = [i for i in degrees if i[0] in category_list]
    degrees = {'keyword': keyword_degrees, 'store': store_degrees, 'category': category_degrees}
    if plot:
        degree_values = [i[1] for i in degrees[degree_type]]
        hist, no_bins = np.histogram(degree_values, bins=50, normed=True)
        center = (no_bins[:-1] + no_bins[1:]) / 2
        lb_centers, lb_counts = bin(degree_values)
        fig = plt.figure()
        plt.scatter(np.log10(center), np.log10(hist), color='b')
        plt.plot(np.log10(lb_centers), np.log10(lb_counts), color='r', linestyle='dashed')
        plt.xlabel('Keyword In-Degree')
        plt.ylabel('Probability')
        fig.show()
        print(lb_centers, lb_counts)
    return keyword_degrees, store_degrees, category_degrees


def bin(data, bin_start=1., first_bin_width=1.4, a=1.6, drop_zeros=True):
    return log_bin.log_bin(data, bin_start, first_bin_width, a, drop_zeros=drop_zeros)












