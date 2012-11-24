
from collections import defaultdict

def frequencies(values):
    """ 
    Returns a list of value-frequency pairs for a given list of elements 
    """
    freqs = defaultdict(int)
    for value in values:
        freqs[value] += 1.0
    return freqs.items()

def majority_vote(data, attribute="target"):
    """ 
    Counts the frequency of different values for an attribute and 
    returns the value with the largets count
    """
    if not data:
        raise ValueError("Can't have a majority of no data")
    freqs = frequencies([x[attribute] for x in data])
    return max(freqs, key=lambda x: x[1])[0]

def _get_values(data, attribute):
    """ Return the set of unique values associated with an attribute """
    return set([x[attribute] for x in data]) 

def get_matching_instances(data, attribute, value):
    """
    Returns a list of examples that have the same value for attribute
    """
    return [x for x in data if x[attribute] == value]

def _choose_attribute(data, attributes, target_attr, fitness_func):
    """
    Iterates through all the attributes and returns the attribute that
    maximises the fitness_func (typically has highest information gain). 
    """
    if not all((data, attributes)):
        raise AttributeError("Can't choose %s from %s" % (attributes, data))
    gain = fitness_func
    gains = [(attr, gain(data, attr, target_attr)) for attr in attributes]
    return max(gains, key=lambda x: x[1])[0]

def create_decision_tree(data, attributes, fitness_func, 
                         tar_attr_name="target", 
                         default_cls_func=majority_vote):
    """
    Build a new decision tree for the records in data.

    All the records in `data` will share the same value for an attribute
    """
    if not data:
        raise ValueError("No data to make tree out of")

    classes = [record[tar_attr_name] for record in data]
    default_class = default_cls_func(data, attribute=tar_attr_name)

    # If there are no more attributes, return the default class.
    if not attributes:
        return default_class

    # Only a single class left, so return that class.
    if len(set(classes)) == 1:
        return classes[0]

    # Otherwise we have some more reduction to do. Choose the next 
    # attribute to split on and make a new tree.
    nxt_attr = _choose_attribute(data, attributes, tar_attr_name, fitness_func)
    tree = {nxt_attr: {}}

    nxt_attr_vals = set([record[nxt_attr] for record in data])
    for value in nxt_attr_vals:
        # recursively create a new sub tree based on the data associated
        # with all the records that have `value` for `nxt_attr`.
        matching_data = get_matching_instances(data, nxt_attr, value)
        attributes = [x for x in attributes if x != nxt_attr]
        subtree = create_decision_tree(matching_data, attributes, 
                                       tar_attr_name=tar_attr_name, 
                                       fitness_func=fitness_func)
        tree[nxt_attr][value] = subtree
    return tree


class DecisionTree(object):

    def _classify_instance(self, instance, tree):
        if isinstance(tree, dict):
            # get the next tree associted with the value the instance
            # has for the next split point.
            split_attr = tree.keys()[0]
            instance_value = instance[split_attr]
            sub_tree = tree[split_attr][instance_value]
            return self._classify_instance(instance, sub_tree)
        # return the leaf classification
        return tree
    
    def classify(self, data):
        """
        Classifies the records provided according to this decision tree.
        """
        return [self._classify_instance(instance) for instance in data ]

    def __init__(self, data, attributes, fitness_func, target_name="target", 
                 default_cls_func=majority_vote):
        self.data = data
        self.attributes = attributes
        self.tree = create_decision_tree(data, attributes, fitness_func, 
                                         tar_attr_name=target_name, 
                                         default_cls_func=default_cls_func)

        