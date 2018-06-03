class Node(object):
	def __init__(self, parent = None, val = None, children = []):
		self.value = val
		self.children = []
		self.parent = parent
		self.height = 0
		self.classification = None
		self.attr_split_index = None
		self.is_leaf = None
		self.attr_val = None
		self.attr_name = None

	@property
	def value(self):
		return self.__value

	@value.setter
	def value(self, val):
		self.__value = val

	def __str__(self):
		return str(self.attr_split_index)