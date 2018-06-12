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
    node = Node(parent=parent_node)#, val=VALUE)

    if(parent_node == None):
        node.height = 0
        node.attr_val = None
    else:
        node.height = node.parent.height + 1
        node.attr_val = parent_attr_val

    # get number of class values containing in dataset
    class_count = count_class_values(dataset.examples)

    # check all examples are not the same type
    # main stop criteria
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
    min_gain = 0.01 # minimum entori
    dataset_entropy = calc_entropy(dataset)

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
        print("Upps. Something went wrong and you should find the mistake, pedik.")
    elif (max_gain <= min_gain or node.height > 20):
        node.is_leaf = True
        # node.classification = classify_leaf(dataset, classifier)
        return node

    node.attr_split_index = attr_to_split
    node.attr_name = dataset.attr_names[attr_to_split]

    # # creating datasets for all values of attributes that is going to be splited
    attr_value_list = [example[attr_to_split] for example in dataset.examples] # these are the values we can split on, now we must find the best one
    attr_value_list = list(set(attr_value_list)) # remove duplicates from list of all attribute values


    for attr_val in attr_value_list:

        new_dataset = Data()
        new_dataset.attr_names = dataset.attr_names

        for example in dataset.examples:
            if(attr_to_split is not None and example[attr_to_split] == attr_val):
                new_dataset.examples.append(example)
        
        node.children.append(compute_tree(new_dataset, node, attr_val))
    return node



# Calculate the entropy of the current dataset
def calc_entropy(dataset):
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
    attr_entropy = 0
    total_examples = len(dataset.examples)

    # count for dataset class
    class_count = count_class_values(dataset.examples)
    new_dataset = Data()
    for example in dataset.examples:
        if(example[attr_index] == val):
            new_dataset.examples.append(example)

    attr_entropy = len(new_dataset.examples)/total_examples * calc_entropy(new_dataset)

    return attr_entropy

def print_tree(node, tab_index):
    if(node.is_leaf):
        print (tab_index * '\t') + "LEAF NODE = ", node.classification, " AND VALUE = ", node.attr_val
        return
    else:
        print (tab_index * '\t') + "Node: ", node.attr_name

    tab_index += 1

    for ch in node.children:
        print_tree(ch, tab_index)

# For testing tree
def fit_tree(node, example):
    if(node.is_leaf):
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


def write_tree(node, tab_index, file):
    if(node.is_leaf):
        file.write((tab_index * '\t') + "LEAF NODE: "  + str(node.classification) + " | PARENT VALUE: " + str(node.attr_val) + '\n')
        return
    else:
        file.write((tab_index * '\t') + "Node: " + str(node.attr_name) + ' | height: ' + str(node.height) + '\n')

    tab_index += 1

    for ch in node.children:
        write_tree(ch, tab_index, file)



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

    dataset.read_attr_data()
    dataset.read_examples_data() 

    # Proportion training set to testing set (1 means only training set)
    PROPORTION = 1

    train_dtset = copy.deepcopy(dataset)
    test_dtset = copy.deepcopy(dataset)
    train_dtset.examples, test_dtset.examples = [], []

    total = len(dataset.examples)
    
    # polluting train dataset
    train_index_list = random.sample(xrange(total), int(total*PROPORTION))
    train_dtset.examples = [ dataset.examples[index] for index in train_index_list if(dataset.examples[index] not in train_dtset.examples)]

    # polluting test dataset
    test_dtset.examples = [ ex for ex in dataset.examples if(ex not in train_dtset.examples)]

    print("Computing tree...")
    root = compute_tree(train_dtset, None, None)
    tree_filename = 'results/tree.txt'
    with open(tree_filename, "w") as tree_file:
        write_tree(root, 0, tree_file)

if __name__ == "__main__":
    main()