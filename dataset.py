import sys
import re


class Data(object):
	def __init__(self, classif = None, attr_num = 6):
		self.attr_names = []
		self.attr_values = []
		self.class_values = []
		self.attr_file = None
		self.data_file = None
		self.attr_number = 6
		self.examples = []


	# reading attributes data from file and assigning to data object
	def read_attr_data(self):
	    file = open(self.attr_file)
	    raw_file = file.read()
	    rowsplit_data = raw_file.splitlines()

	    # where attributes start in file
	    attr_row_start = 0

	    # start with class values
	    for index, row in enumerate(rowsplit_data):
	        if(row == '| class values'):
	            index += 2
	            self.class_values = re.split(',|:', rowsplit_data[index])
	            break

	    # start with attributes index
	    for index, row in enumerate(rowsplit_data):
	        if(row == '| attributes'):
	            attr_row_start = index+2

	    # start with getting attributes
	    for attr in rowsplit_data[attr_row_start:attr_row_start+self.attr_number]:
	        row = [x.replace(' ', '').replace(".",'') if(' ' in x) else x.replace('.','') for x in re.split(':|,', attr)]
	        self.attr_names.append(row[0])
	        self.attr_values.append(row[1:])


	# reading examples data from file and assigning to data object
	def read_examples_data(self):
	    file = open(self.data_file)
	    raw_file = file.read()
	    rowsplit_data = raw_file.splitlines()
	    
	    for rows in rowsplit_data:
	        data_row = rows.split(',')
	        # getting data examples
	        self.examples.append(data_row)



	def read_from_ball_file(self):
	    print("Reading data...")
	    f = open(self.data_file)
	    original_file = f.read()
	    rowsplit_data = original_file.splitlines()
	    self.examples = [rows.split(',') for rows in rowsplit_data]

	    #list attributes
	    self.attr_names = self.examples.pop(0)