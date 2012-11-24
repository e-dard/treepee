
import math

from dtree import frequencies, get_matching_instances

""" functions associated with the ID3 algorithm """



def _entropy(data, target_attr):
    """
    Calculates the entropy of the given data set, which should be of 
    the form of a distribution of values.
    """
    value_dist = frequencies([instance[target_attr] for instance in data])
    # calculate the entropy of the data associated with `attribute`
    sum_frequencies = float(sum([x[1] for x in value_dist]))
    data_entropy = 0
    for _, frequency in value_dist:
        p_value = frequency / sum_frequencies
        data_entropy += (-p_value * math.log(p_value, 2))
    return data_entropy


def information_gain(data, split_attr, target_attr):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on `split_attr`
    """
    data_entropy = _entropy(data, target_attr)
    spl_att_val_dist = frequencies([instance[split_attr] for instance in data])
    total_subset_entropy = 0.0

    # For each value in the attribute we are splitting on get the 
    # remaining entropy (in terms of the instance class) of the subset 
    # with that value
    for value, value_freq in spl_att_val_dist:
        p_value = value_freq / float(len(data))
        data_subset = get_matching_instances(data, split_attr, value)
        total_subset_entropy += p_value * _entropy(data_subset, target_attr)
    
    # Subtract the subset entropy after we split on `split_attr` from 
    # the current data set entropy, giving us the information gain from
    # splitting on `split_attr`
    return data_entropy - total_subset_entropy

