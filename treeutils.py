from __future__ import print_function

from sklearn.metrics import make_scorer, f1_score
from sklearn.tree import _tree
import numpy as np, random as rnd

import sys
try:
    from cStringIO import StringIO
except:
    from io import StringIO

def simplify_tree(decision_tree, X, y, scorer=make_scorer(f1_score, greater_is_better=True), acceptable_score_drop=0.0, verbose=1):
    current_score, original_score = 0, 1
    
    while current_score != original_score:
        current_score = scorer(decision_tree, X, y)
        original_score = current_score
        tree = decision_tree.tree_

        removed_branches = []
        nodes = range(tree.node_count)
        rnd.shuffle(nodes)
        for i in nodes:
            current_left, current_right = tree.children_left[i], tree.children_right[i]

            if tree.children_left[i] >= 0 or tree.children_right[i] >= 0:
                tree.children_left[i], tree.children_right[i] = -1, -1
                auc = scorer(decision_tree, X, y)
                if auc >= current_score - acceptable_score_drop:
                    current_score = auc
                    removed_branches.append(i)
                else:
                    tree.children_left[i], tree.children_right[i] = current_left, current_right

        if verbose:
            print("Removed",len(removed_branches)," branches. current score: ", current_score)
        
    return decision_tree

def tree_to_code(tree, feature_names, decimals=4, transform_to_probabilities=True):
    tree_ = tree.tree_
    tree_feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rounding_multiplier = np.power(10, decimals)
    round = lambda x: np.round(x*rounding_multiplier)/rounding_multiplier
    def leaf_value(value, samples=1):
        if transform_to_probabilities:
            return round(value / samples)[0][1]
        else:
            return value[0]
    
    stdout_ = sys.stdout
    sys.stdout = StringIO()
    
    print("def probability_of_class_one({}):".format(", ".join(feature_names))+"")

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = tree_feature_name[node]
            threshold = tree_.threshold[node]
            
            if tree_.feature[tree_.children_left[node]] == _tree.TREE_UNDEFINED and \
              tree_.feature[tree_.children_right[node]] == _tree.TREE_UNDEFINED and \
              np.all(np.equal(tree_.value[tree_.children_left[node]], tree_.value[tree_.children_right[node]])):
                print("{}return {}".format(indent, leaf_value(tree_.value[node], tree_.weighted_n_node_samples[node])))
                 
            else:
                print("{}if {} <= {}:".format(indent, name, round(threshold)))
                recurse(tree_.children_left[node], depth + 1)
                print("{}else:".format(indent)) #  # if {} > {}".format(indent, name, threshold))
                recurse(tree_.children_right[node], depth + 1)
        else:
            if transform_to_probabilities:
                p = round(tree_.value[node] / tree_.weighted_n_node_samples[node])[0]
            else:
                p = tree_.value[node]
                
            print("{}return {}".format(indent, leaf_value(tree_.value[node], tree_.weighted_n_node_samples[node])))

    recurse(0, 1)

    string = sys.stdout.getvalue()
    sys.stdout = stdout_
    return string