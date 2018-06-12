class Node(object):
	def __init__(self, parent = None, children = []):
		self.children = []
		self.parent = parent
		self.height = 0
		self.classification = None
		self.attr_split_index = None
		self.is_leaf = None
		self.attr_val = None
		self.attr_name = None

	def __str__(self):
		return str(self.attr_name)