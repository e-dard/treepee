
from collections import defaultdict

def frequencies(values):
    """ 
    Returns a list of value-frequency pairs for a given list.

    :param values: an iterable containing values to count.

    :returns: a list of the form [(value, frequency), ...] The list is 
    unsorted. 
    """
    freqs = defaultdict(int)
    for value in values:
        freqs[value] += 1.0
    return freqs.items()

def majority_vote(data, attribute="target"):
    """ 
    Returns the mode for values associated with `attribute` in `data`.

    :param data: an iterable of data intances

    :param attribute: optional argument specifying which attribute to 
                      examine for values. By default this is "target", 
                      which assumes the target attribute is named 
                      accordingly.

    :returns: the mode for the values associated with `attribute`.
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
    Filter instances in `data` that don't havae `value` for `attribute`.

    :param data: iterable dataset to filter on.

    :param attribute: attreibute to check for each instance in `data`.

    :param value: value for `attribute` that instance must have to be 
                  included in the resulting list.

    :returns: a list of instances in `data` where each `attribute` has 
              `value`.
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
    Build a new decision based upon the example instances in `data`.

    This function recursively builds a decision tree by choosing the 
    attribute to split the tree on, based upon some splitting criteria 
    (usually whatever maximises information gain). Eventually, all the 
    remaining instances passed in will have the same output class, and 
    therefore a leaf node containing that class is returned.

    :param data: an iterable containing instances to build the tree on.

    :param attributes: an iterable describing the labels for each 
                       attribute in `data`. **Note** this iterable 
                       should not contain the target attribute label.
    
    :param fitness_func: a function that should accept a dataset, 
                         attribute label, and target attribute label, 
                         and should return the fitness (usually gain) 
                         from splitting `data` on that attribute.

    :param tar_attr_name: optional argument specifying the label 
                          associated with the classification for an 
                          instance. Defaults to 'target'.

    :param default_cls_func: optional argument specifying function to 
                             determine the default class associated 
                             with an instance if no attributes are 
                             available to split on. Defaults to a 
                             majority vote.

    :returns: a new decision tree.
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
        """ return the classification for `instance` using tree. """
        if isinstance(tree, dict):
            # get the next tree associted with the value the instance
            # has for the next split point.
            split_attr = tree.keys()[0]
            instance_value = instance[split_attr]
            sub_tree = tree[split_attr][instance_value]
            return self._classify_instance(instance, sub_tree)
        # return the leaf classification
        return (instance, tree)
    
    def classify(self, data):
        """
        Classifies the instances provided using the decision tree.

        :param data: an iterable of instances to classify

        :returns a new list of tuples containing in the first element 
                 each original instance from `data` and in the second 
                 element the classification.
        """
        return [self._classify_instance(instance, self.tree) \
               for instance in data]

    def __init__(self, data, attributes, fitness_func, target_name="target", 
                 default_cls_func=majority_vote):
        self.data = data
        self.attributes = attributes
        self.tree = create_decision_tree(data, attributes, fitness_func, 
                                         tar_attr_name=target_name, 
                                         default_cls_func=default_cls_func)

        