import numpy as np
from collections import Counter

""" ------------------- helper functions below ------------------- """


# extract and return the features and labels of instances from dataset
def read_dataset(filepath):
    x, y_lables = [], []
    with open(filepath) as data_file:
        for line in data_file:
            if line.strip() != "":
                row = line.strip().split(',')
                x.append(list(map(int, row[0:-1])))
                y_lables.append(row[-1])

    x = np.array(x)
    y_lables = np.array(y_lables)
    return x, y_lables


# get accuracy by comparing the ground truth and predictions
def get_accuracy(y_gold, y_predictions):
    assert len(y_gold) == len(y_predictions)
    try:
        return np.sum(y_predictions == y_gold) / len(y_gold)
    except ZeroDivisionError:
        return 0


# get the information entropy from the labels of the instances
def get_information_entropy(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)

    p, ie = 0, 0  # probability, information entropy
    for label, count in zip(unique_labels, counts):
        p = count / len(labels)
        ie += (-p) * np.log2(p)
    return ie


# split the attribute and value that provides the best information gain
def split(instances, labels):
    ig_max, split_attribute, split_value = 0, None, None
    entropy_prev = get_information_entropy(labels)
    num_of_attributes = instances.size // len(instances)

    for attribute in range(num_of_attributes):
        min_value = instances.min(axis=0)[attribute]
        max_value = instances.max(axis=0)[attribute]

        for value in range(min_value, max_value + 1):
            left_index = instances[:, attribute] <= value
            right_index = instances[:, attribute] > value
            left_labels = labels[left_index]
            right_labels = labels[right_index]

            # calculate information gain
            entropy_left = len(left_labels) / len(labels) * get_information_entropy(left_labels)
            entropy_right = len(right_labels) / len(labels) * get_information_entropy(right_labels)
            ig = entropy_prev - (entropy_left + entropy_right)

            if ig > ig_max:
                # Note: if ig >= ig_max, then accuracy = 0.91
                # update ig_max, attribute and value that generates the largest information gain
                ig_max = ig
                split_attribute = attribute
                split_value = value

    return split_attribute, split_value


# return the prediction of the instance using the model
def predict_output(node, instances):
    # base case
    if node.is_leaf:
        if len(np.unique(node.labels)) == 1:
            return node.labels[0]
        else:
            # this condition happens when a node is pruned where
            # more than one labels is stored in that particular node
            # for that, pick the most frequent labels among the instances
            return list(dict(Counter(node.labels).most_common(1)))[0]

    if instances[node.split_attribute] <= node.split_value:
        return predict_output(node.left, instances)
    else:
        return predict_output(node.right, instances)


# get the max depth of the tree
def get_max_depth(root):
    if root is None:
        return 0
    left_depth = get_max_depth(root.left)
    right_depth = get_max_depth(root.right)
    return max(left_depth, right_depth) + 1  # 1 is the root


# attempts to prune the tree
def explore_nodes_to_prune(classifier, node, x_val, y_val, depth_to_explore):
    # keep explore the depth of the tree until it reaches the depth we want to explore
    if node.left:
        if depth_to_explore > node.depth:
            explore_nodes_to_prune(classifier, node.left, x_val, y_val, depth_to_explore)
    if node.right:
        if depth_to_explore > node.depth:
            explore_nodes_to_prune(classifier, node.right, x_val, y_val, depth_to_explore)

    if node.depth != depth_to_explore:  # skip where node depth is not the depth we want to explore
        return
    if node.is_leaf:  # skip any leaf node as pruning that has no meaning
        return

    # possible conditions to impose pruning: when one of its child is a leaf node
    if node.left.is_leaf or node.right.is_leaf:

        prune_acc_prev = get_accuracy(y_val, classifier.predict(x_val))
        node.is_leaf = True  # pruning
        prune_acc_after = get_accuracy(y_val, classifier.predict(x_val))

        # if accuracy does not improve, undo the pruning
        # else, doing nothing which keeps the node pruned
        if prune_acc_after <= prune_acc_prev:
            node.is_leaf = False


""" ----------------- class Node implementation below ----------------- """


class Node:
    def __init__(self, instances, labels, depth, split_attribute=None, split_value=None):
        self.instances = instances
        self.labels = labels
        self.depth = depth
        self.split_attribute = split_attribute
        self.split_value = split_value
        self.left = None
        self.right = None
        self.is_leaf = False

    def insert_left_child(self, index_list):
        self.left = Node(self.instances[index_list], self.labels[index_list], self.depth + 1)

    def insert_right_child(self, index_list):
        self.right = Node(self.instances[index_list], self.labels[index_list], self.depth + 1)


""" ----------------- class Decision Tree Classifier implementation below ----------------- """


class DecisionTreeClassifier(object):
    """ Basic decision tree classifier
    
    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    
    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """

    def __init__(self):
        self.root = None
        self.is_trained = False

    def fit(self, x, y):
        """ Constructs a decision tree classifier from data
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (N, K) 
                           N is the number of instances
                           K is the number of attributes
        y (numpy.ndarray): Class labels, numpy array of shape (N, )
                           Each element in y is a str 
        """

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        self.root = Node(x, y, depth=0)
        self.induce_decision_tree(self.root)

        self.is_trained = True

    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.
        
        Assumes that the DecisionTreeClassifier has already been trained.
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (M, K) 
                           M is the number of test instances
                           K is the number of attributes
        
        Returns:
        numpy.ndarray: A numpy array of shape (M, ) containing the predicted
                       class label for each instance in x
        """

        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")

        # set up an empty (M, ) numpy array to store the predicted labels
        predictions = np.zeros((x.shape[0],), dtype=np.object)
        for j, instance in enumerate(x):
            predictions[j] = predict_output(self.root, instance)

        return predictions

    def prune(self, x_val, y_val):
        """ Post-prune your DecisionTreeClassifier given some optional validation dataset.

        You can ignore x_val and y_val if you do not need a validation dataset for pruning.

        Args:
        x_val (numpy.ndarray): Instances of validation dataset, numpy array of shape (L, K).
                           L is the number of validation instances
                           K is the number of attributes
        y_val (numpy.ndarray): Class labels for validation dataset, numpy array of shape (L, )
                           Each element in y is a str 
        """

        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")

        # get the maximum depth
        deepest_depth = get_max_depth(self.root)

        # explore the depth starting from (max_depth - 1) to half of the max_depth
        half_of_max_depth = deepest_depth // 2
        for depth in range(deepest_depth - 1, half_of_max_depth, -1):
            explore_nodes_to_prune(self, self.root, x_val, y_val, depth)

        print("Pruning completed")

    # induce the decision tree
    def induce_decision_tree(self, root):
        # obtain instances and labels
        instances = root.instances
        labels = root.labels

        if len(np.unique(labels)) == 1:
            root.is_leaf = True
            return

        # save the split attribute and split value at this node
        split_attribute, split_value = split(instances, labels)
        root.split_attribute = split_attribute
        root.split_value = split_value

        left_labels = instances[:, split_attribute] <= split_value
        right_labels = instances[:, split_attribute] > split_value

        # construct left and right child
        root.insert_left_child(left_labels)
        root.insert_right_child(right_labels)

        # construct the tree recursively
        self.induce_decision_tree(root.left)
        self.induce_decision_tree(root.right)
