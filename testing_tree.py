from __future__ import division
from tree import Node
from dataset import Data, read_examples_data, read_attr_data
from creating_tree import compute_tree, fit_tree
from tree import Node

import sys
import unittest
import random
import copy

class TestStringMethods(unittest.TestCase):

    def setUp(self):
        self.filename1 = 'car.c45-names.txt' #attributes
        self.filename2 = 'car.data' # data examples

        self.dataset = Data()
        self.dataset.attr_file = self.filename1
        self.dataset.data_file = self.filename2

        self.dataset.read_attr_data()
        self.dataset.read_examples_data()

    def test_fittin_tree(self):
        root = compute_tree(self.dataset, None, None)
        val = 0
        for ex in self.dataset.examples:
            val += 1
            ex_data = ex[:len(ex)-1]
            origin_class = ex[len(ex)-1]
            tree_classif = fit_tree(root, ex_data)

            self.assertEqual(tree_classif, origin_class)

    def test_fittin_not_full_tree(self):
        trues = 0
        falses = 0
        loc_dataset = copy.deepcopy(self.dataset)

        loc_dataset.examples = loc_dataset.examples[::2]
        # print(len(loc_dataset.examples))
        root = compute_tree(loc_dataset, None, None)
        # print 'root: ', root
        val = 0
        
        for ex in self.dataset.examples:
            val += 1
            ex_data = ex[:len(ex)-1]
            origin_class = ex[len(ex)-1]
            tree_classif = fit_tree(root, ex_data)

            if(tree_classif == origin_class):
                trues += 1
            else:
                falses += 1
        # print(trues, val)
        self.assertNotEqual(trues, len(self.dataset.examples))


    def test_fit_func_ranking(self):
        # training dataset
        train_dtset = Data()
        # testing dataset
        test_dtset = Data()
        train_dtset.attr_file = self.filename1
        test_dtset.attr_file = self.filename1

        # attributes data
        train_dtset.read_attr_data()
        test_dtset.read_attr_data()

        # poluting training
        # train to test = 80/20
        # random list from 0 to (length of ex - 1)
        train_index_list = random.sample(xrange(len(self.dataset.examples)), int(len(self.dataset.examples)*0.8))
        train_dtset.examples = [ self.dataset.examples[index] for index in train_index_list if(self.dataset.examples[index] not in train_dtset.examples)]

        test_dtset.examples = [ ex for ex in self.dataset.examples if(ex not in train_dtset.examples)]

        root = compute_tree(train_dtset, None, None)

        print test_examples(root, test_dtset)


def test_examples(node, dtset):
    trues = 0
    for ex in dtset.examples:
        ex_data = ex[:-1]
        origin_class = ex[-1]
        tree_classif = fit_tree(node, ex_data)

        if(tree_classif == origin_class):
            trues += 1

    return trues/len(dtset.examples)


def main():

    unittest.main()


if __name__ == "__main__":
    main()