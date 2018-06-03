from __future__ import division
from tree import Node
from dataset import Data
from tree import Node

import sys
import re
from collections import Counter
import math
import copy
import random


VALUE = 0

# reading attributes data from file and assigning to data object
def read_attr_data(dataset):
    file = open(dataset.attr_file)
    raw_file = file.read()
    rowsplit_data = raw_file.splitlines()

    # where attributes start in file
    attr_row_start = 0

    # start with class values
    for index, row in enumerate(rowsplit_data):
        if(row == '| class values'):
            index += 2
            dataset.class_values = re.split(',|:', rowsplit_data[index])
            break

    # start with attributes index
    for index, row in enumerate(rowsplit_data):
        if(row == '| attributes'):
            attr_row_start = index+2

    # start with getting attributes
    for attr in rowsplit_data[attr_row_start:attr_row_start+dataset.attr_number]:
        row = [x.replace(' ', '').replace(".",'') if(' ' in x) else x.replace('.','') for x in re.split(':|,', attr)]
        dataset.attr_names.append(row[0])
        dataset.attr_values.append(row[1:])


# reading examples data from file and assigning to data object
def read_examples_data(dataset):
    file = open(dataset.data_file)
    raw_file = file.read()
    rowsplit_data = raw_file.splitlines()
    
    for rows in rowsplit_data:
        data_row = rows.split(',')
        # getting data examples
        dataset.examples.append(data_row)



def read_from_ball_file(dataset):
    print "Reading data..."
    f = open(dataset.data_file)
    original_file = f.read()
    rowsplit_data = original_file.splitlines()
    dataset.examples = [rows.split(',') for rows in rowsplit_data]

    #list attributes
    dataset.attr_names = dataset.examples.pop(0)


def count_class_values(examples):
    # print(indx)
    class_values = [ex[-1] for ex in examples]
    class_values = Counter(class_values)
    # print(class_values)
    class_values_count = []
    for class_indx in range(len(class_values)):
        class_values_count.append(class_values.most_common()[class_indx])

    return class_values_count 

def compute_tree(dataset, parent_node, parent_attr_val):
    global VALUE

    node = Node(parent=parent_node, val=VALUE)

    VALUE += 1

    if(parent_node == None):
        node.height = 0
        node.attr_val = None
    else:
        node.height = node.parent.height + 1
        node.attr_val = parent_attr_val

    # get number of class values containing in dataset
    class_count = count_class_values(dataset.examples)

    # check all examples are not the same type
    for indx in range(len(class_count)):
        if(len(dataset.examples) == class_count[indx][1]):
            node.classification = class_count[indx][0]
            node.is_leaf = True
            return node
        else:
            node.is_leaf = False

    attr_to_split = None # The index of the attribute we will split on
    max_gain = 0 # The gain given by the best attribute
    split_val = None 
    min_gain = 0.01
    dataset_entropy = calc_entropy(dataset, dataset.class_index)

    for attr_index in range(len(dataset.examples[0])-1):

        # if(dataset.attr_names[attr_index] != dataset.attr_names[len(dataset.attr_names) - 1]):

        local_max_gain = 0
        local_split_val = None
        attr_value_list = [example[attr_index] for example in dataset.examples] # these are the values we can split on, now we must find the best one
        attr_value_list = list(set(attr_value_list)) # remove duplicates from list of all attribute values

        # for each attribute go through value to get overall information gain
        local_gain = 0
        for val in attr_value_list:
            local_gain += calc_gain(dataset, dataset_entropy, val, attr_index) # calculate the gain if we split on this value

        local_gain = dataset_entropy - local_gain

        if(local_gain > max_gain):
            max_gain = local_gain
            attr_to_split = attr_index

    #attr_to_split is now the best attribute according to our gain metric
    if (attr_to_split is None):
        print "Upps. Something went wrong and you should find the mistake, pedik."
    elif (max_gain <= min_gain or node.height > 20):
        node.is_leaf = True
        node.classification = classify_leaf(dataset, classifier)

        return node

    node.attr_split_index = attr_to_split
    node.attr_name = dataset.attr_names[attr_to_split]

    # # creating datasets for all values of attributes that is going to be splited
    attr_value_list = [example[attr_to_split] for example in dataset.examples] # these are the values we can split on, now we must find the best one
    attr_value_list = list(set(attr_value_list)) # remove duplicates from list of all attribute values

    if(VALUE == 1):
        attr_val = attr_value_list[1]
    else:
        attr_val = attr_value_list[0]

    # print(dataset.attr_names[attr_to_split])

    for attr_val in attr_value_list:

        new_dataset = Data()
        new_dataset.attr_names = dataset.attr_names

        for example in dataset.examples:
            if(attr_to_split is not None and example[attr_to_split] == attr_val):
                new_dataset.examples.append(example)
        
        node.children.append(compute_tree(new_dataset, node, attr_val))

    return node



# Calculate the entropy of the current dataset
def calc_entropy(dataset, attr_index):
    class_count = count_class_values(dataset.examples)
    total_examples = len(dataset.examples)

    entropy = 0
    for class_indx in range(len(class_count)):
        p = class_count[class_indx][1] / total_examples
        if (p != 0):
            entropy += p * math.log(p, 2)

    entropy = -entropy
    return entropy

# Calculate the gain of a attribute split
def calc_gain(dataset, entropy, val, attr_index):
    classifier = dataset.attr_names[attr_index]
    attr_entropy = 0
    total_examples = len(dataset.examples)

    # count for dataset class
    class_count = count_class_values(dataset.examples)
    new_dataset = Data()
    for example in dataset.examples:
        if(example[attr_index] == val):
            new_dataset.examples.append(example)

    attr_entropy = len(new_dataset.examples)/total_examples * calc_entropy(new_dataset, attr_index)

    return attr_entropy


def print_tree(node):
    if(node.is_leaf):
        print("leaf node = ", node.classification, " and class_value = ", node.attr_val)
        return
    else:
        print("not leaf node = ", node.attr_name)

    for ch in node.children:
        print_tree(ch)

# For testing tree
def fit_tree(node, example):
    if(node.is_leaf):
        # print node.height
        # print 'Leaf node: ', node.classification
        return node.classification

    cur_node_attr_index = node.attr_split_index
    ex_attr_val = example[cur_node_attr_index]

    for child in node.children:
        if(ex_attr_val == child.attr_val):
            classif = fit_tree(child, example) 
            return classif

# counting test dataset examples guessed
def test_examples(node, dtset):
    trues = 0
    for ex in dtset.examples:
        ex_data = ex[:-1]
        origin_class = ex[-1]
        tree_classif = fit_tree(node, ex_data)

        if(tree_classif == origin_class):
            trues += 1

    return trues/len(dtset.examples)



# choosing random examples from training dataset
def random_examples_dtset(train_dtset, num_subset):
    random_dtset = copy.deepcopy(train_dtset)
    random_dtset.examples = []

    total = len(train_dtset.examples)
    # polluting train dataset
    random_set_index_list = random.sample(xrange(total), num_subset)
    random_dtset.examples = [ train_dtset.examples[index] for index in random_set_index_list]

    return random_dtset

def rank_with_counting_appearances(attr_count, ranking_lst, attr_names):
    # adding to ranking list if attribute appears in the tree
    for attr in attr_names:
        if(attr in attr_count):
            ranking_lst[attr] += 1

def rank_with_counting_number_of_appeance_in_each_tree(attr_count_num, ranking_lst_cnt, attr_names):
    # adding to ranking list if attribute appears in the tree
    for attr in attr_names:
        ranking_lst_cnt[attr] += attr_count_num[attr]

# ranking function
def processing_for_ranking(train_dtset, ranking_lst, N, ranking_lst_count):
    # getting random dataset with N elements
    random_subset = random_examples_dtset(train_dtset, N)

    # growing tree on random set of data
    root = compute_tree(random_subset, None, None)

    attr_height = {}
    attr_count_num = {}
    for name in train_dtset.attr_names:
        attr_count_num[name] = 0

    ranking(root, attr_height, train_dtset.attr_names, attr_count_num)
    rank_with_counting_appearances(attr_height, ranking_lst, train_dtset.attr_names)
    rank_with_counting_number_of_appeance_in_each_tree(attr_count_num, ranking_lst_count, train_dtset.attr_names)

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
# argument must be counter most_common
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



def main():
    args = sys.argv
    if(len(args) < 2):
        print("You should provide a filename to data.")
        filename1 = 'car.c45-names.txt' #attributes
        filename2 = 'car.data' # data examples
    else:
        filename1 = str(sys.argv[0])
        filename2 = str(sys.argv[1]) # data examples

    dataset = Data()
    dataset.attr_file = filename1
    dataset.data_file = filename2

    dataset1 = Data(attr_num = 4)
    dataset1.data_file = 'ball.txt'

    # read_from_ball_file(dataset1)

    dataset.read_attr_data()
    dataset.read_examples_data() 

    # cars tree
    print("Computing tree...")
    root = compute_tree(dataset, None, None)

    # ball tree
    # root1 = compute_tree(dataset1, None, None)
    ranking_list = Counter()
    ranking_list_counting = Counter()
    for attr in dataset.attr_names:
        ranking_list[attr] = 0
        ranking_list_counting[attr] = 0

    ranking_list_final = {}

    # repetition of tree creation
    M = 10
    # subset length of random elements from testing subset
    N = 100
    # Proportion training set to testing set
    PROPORTION = 0.8

    train_dtset = copy.deepcopy(dataset)
    test_dtset = copy.deepcopy(dataset)
    train_dtset.examples, test_dtset.examples = [], []

    total = len(dataset.examples)
    # polluting train dataset
    train_index_list = random.sample(xrange(total), int(total*PROPORTION))
    train_dtset.examples = [ dataset.examples[index] for index in train_index_list if(dataset.examples[index] not in train_dtset.examples)]

    # polluting test dataset
    test_dtset.examples = [ ex for ex in dataset.examples if(ex not in train_dtset.examples)]

    for i in range(M):
        processing_for_ranking(train_dtset, ranking_list, N, ranking_list_counting)

    print("Count of elements in {0} element subset for {1} times.".format(N, M))
    print(ranking_list.most_common())
    print(ranking_list_counting.most_common())

    print(final_rank(ranking_list.most_common()))
    print(final_rank(ranking_list_counting.most_common()))

    # counting the most common attribute height in the tree and by calcing 
    # for val in dataset.attr_names:
    #     sum = 0
    #     ls = []
    #     for ex in ranking_list:
    #         sum += ex[val]
    #         ls.append(ex[val])

    #     ranking_list_final[val] = sum / 10
    #     ranking_list_final[val] = Counter(ls).most_common()[0][0]

    # print(ranking_list_final)

    

if __name__ == "__main__":
    main()