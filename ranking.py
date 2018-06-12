from __future__ import division

from tree import Node
from dataset import Data
from creating_tree import compute_tree

import sys
import re
from collections import Counter
import collections
import math
import copy
import random
import pandas as pd

import matplotlib.pyplot as plt


# choosing random examples from training dataset
def random_examples_dtset(train_dtset, num_subset):
    random_dtset = copy.deepcopy(train_dtset)
    random_dtset.examples = []

    total = len(train_dtset.examples)
    # polluting train dataset
    random_set_index_list = random.sample(range(total), num_subset)
    random_dtset.examples = [ train_dtset.examples[index] for index in random_set_index_list]

    return random_dtset

def rank_with_counting_number_of_appeance_in_each_tree(attr_count_num, ranking_lst_cnt, attr_names):
    # adding to ranking list if attribute appears in the tree
    for attr in attr_names:
        ranking_lst_cnt[attr] += attr_count_num[attr]

# ranking function
def processing_for_ranking(train_dtset, N, ranking_lst_count):
    # getting random dataset with N elements
    random_subset = random_examples_dtset(train_dtset, N)

    # growing the tree on random set of data
    root = compute_tree(random_subset, None, None)
    # print('New tree !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # print_tree(root, 0)

    attr_height = {}
    attr_count_num = {}
    for name in train_dtset.attr_names:
        attr_count_num[name] = 0

    ranking(root, attr_height, train_dtset.attr_names, attr_count_num)
    # rank_with_counting_appearances(attr_height, ranking_lst, train_dtset.attr_names)
    rank_with_counting_number_of_appeance_in_each_tree(attr_count_num, ranking_lst_count, train_dtset.attr_names)


def print_tree(node, tab_index):
    if(node.is_leaf):
        print((tab_index * '\t') + "LEAF NODE = ", node.classification, " AND VALUE = ", node.attr_val)
        return
    else:
        print((tab_index * '\t') + "Node: ", node.attr_name)

    tab_index += 1

    for ch in node.children:
        print_tree(ch, tab_index)

# RANK with counter for most common number
def ranking(node, count, attr_names, attr_count):
    if(node.is_leaf):
        return

    cur_node_attr_index = node.attr_split_index
    # adding current attribute
    # if(str(attr_names[cur_node_attr_index]) not in count):
    count[str(attr_names[cur_node_attr_index])] = node.height
    attr_count[str(attr_names[cur_node_attr_index])] += 1

    for child in node.children:
        ranking(child, count, attr_names, attr_count) 

# rank attributes with their most appearence in M random trees
# argument must be Counter most_common
# assigns each attribute the rank number
def final_rank(rank_list):
    rank = {}
    count = 1

    val_idx = 0
    while(val_idx < len(rank_list)):
        best_val = rank_list[val_idx]
        rank[best_val[0]] = count

        for val1_idx, val1 in enumerate(rank_list):
            if(val1[1] == best_val[1]):
                rank[val1[0]] = count
                val_idx += 1

        count += 1

    return rank


def diagram_printing_num(data_list, filename, tit, ylab):
    rank_keys = [elem for elem in collections.OrderedDict(sorted(data_list[0].items()))]

    # preparing data points for pandas
    data_points = []
    # sorting by key
    data_list = [ collections.OrderedDict(sorted(data.items())) for data in data_list]
    for ex in data_list:
        data_points.append([ value for key,value in ex.items()])

    df2 = pd.DataFrame(data_points, columns=rank_keys, index=range(1,len(data_points)+1))
    df2.index.name = 'NM'
    print(df2)

    df2.to_csv(filename, encoding='utf-8')
    df2.plot.bar(title=tit)
    plt.ylabel(ylab)
    plt.show()

def main():
    args = sys.argv

    filename1 = 'car.c45-names.txt' #attributes
    filename2 = 'car.data' # data examples

    # repetition of tree creation
    M = 10
    # subset length of random elements from testing subset
    N = 800
    # repeatition of M time creation of tree
    NM = 10
    # Proportion training set to testing set
    PROPORTION = 0.9
        
    if ("-n" in args):
        try:
            N = int(args[args.index("-n") + 1])
        except Exception as e:
            print(e)
            return 
    else:
        print('Parametr for N is not defined.')
        return

    if("-m" in args):
        try:
            M = int(args[args.index("-m") + 1])
        except Exception as e:
            print(e)
            return 
    else:
        print('Parametr for M is not defined.')
        return

    if("-nm" in args):
        try:
            NM = int(args[args.index("-nm") + 1])
        except Exception as e:
            print(e)
            return 
    else:
        print('Parametr for NM is not defined.')
        return


    dataset = Data()
    dataset.attr_file = filename1
    dataset.data_file = filename2

    dataset.read_attr_data()
    dataset.read_examples_data() 

    print("Computing tree...")

    # counts each appearance of attr in each tree
    ranking_list_counting = Counter()

    for attr in dataset.attr_names:
        # ranking_list[attr] = 0
        ranking_list_counting[attr] = 0

    # counting ranking
    ranking_list_final = []
    num_attributes = []

    # running method for ranking_list_counting, means that counts number of appearences in each tree
    for i in range(NM):
        ranking_list_counting = Counter({key: 0 for (key,val) in ranking_list_counting.items()})

        for i in range(M):
            # function that includes creating tree
            processing_for_ranking(dataset, N, ranking_list_counting)

        num_attributes.append(ranking_list_counting)
        ranking_list_final.append(final_rank(ranking_list_counting.most_common()))

    title_rank = 'Ranking atrybutów drzewa wykonujacego sie '+str(M)+' razy i rozmiar podzbioru losowego ' + str(N)
    y_label = "Ranking atrytutów (1 - najczęsciej pojawiający się)"
    filename = 'results/ranking_N'+str(N)+'_M'+str(M)+'.csv'
    print("Ranking attributes for {0} tree repetations and {1} length of random subset.".format(M, N))
    diagram_printing_num(ranking_list_final, filename,title_rank, y_label)

    title_count = 'Ilość atrybutów w drzewie` wykonujacego sie '+str(M)+' razy i rozmiar podzbioru losowego ' + str(N)
    y_label = "Ilość pojawianie się atr. (im wyższy - tym więcej)"
    filename = 'results/attrs_number_N'+str(N)+'_M'+str(M)+'.csv'
    print("Ranking attributes for {0} tree repetations and {1} length of random subset.".format(M, N))
    diagram_printing_num(num_attributes, filename, title_count, y_label)



if __name__ == "__main__":
    main()