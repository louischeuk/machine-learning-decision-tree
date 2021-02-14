from classification import *
from numpy.random._generator import default_rng


# calculate and return the confusion matrix
def confusion_matrix(y_gold, y_prediction, class_labels):
    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

    # for each correct class (row),
    # compute how many instances are predicted for each class (columns)
    for (i, label) in enumerate(class_labels):
        # get predictions where the ground truth is the current class label
        indices = (y_gold == label)
        gold = y_gold[indices]
        predictions = y_prediction[indices]

        # quick way to get the counts per label
        (unique_labels, counts) = np.unique(predictions, return_counts=True)

        # convert the counts to a dictionary
        frequency_dict = dict(zip(unique_labels, counts))

        # fill up the confusion matrix for the current row
        for (j, class_label) in enumerate(class_labels):
            confusion[i, j] = frequency_dict.get(class_label, 0)

    return confusion


# get accuracy form the confusion matrix
def accuracy_from_confusion(confusion):
    if np.sum(confusion) > 0:
        return np.sum(np.diag(confusion) / np.sum(confusion))
    else:
        return 0


# calculate and return the recall and marco-average recall
def recall(y_gold, y_prediction, class_labels):
    confusion = confusion_matrix(y_gold, y_prediction, class_labels)
    r = np.zeros((len(confusion),))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[c, :]) > 0:
            r[c] = confusion[c, c] / np.sum(confusion[c, :])

    macro_r = 0.
    if len(r) > 0:
        macro_r = np.mean(r)

    return r, macro_r


# calculate and return the precision and marco-average precision
def precision(y_gold, y_prediction, class_labels):
    confusion = confusion_matrix(y_gold, y_prediction, class_labels)
    p = np.zeros((len(confusion),))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[:, c]) > 0:
            p[c] = confusion[c, c] / np.sum(confusion[:, c])

    macro_p = 0.
    if len(p) > 0:
        macro_p = np.mean(p)

    return p, macro_p


# calculate and return the F1 scores and marco-average F1 scores
def f1_score(y_gold, y_prediction, class_labels):
    (precisions, macro_p) = precision(y_gold, y_prediction, class_labels)
    (recalls, macro_r) = recall(y_gold, y_prediction, class_labels)

    assert len(precisions) == len(recalls)

    f = np.zeros((len(precisions),))
    for c, (p, r) in enumerate(zip(precisions, recalls)):
        if p + r > 0:
            f[c] = 2 * p * r / (p + r)

    marco_f = 0
    if len(f) > 0:
        marco_f = np.mean(f)

    return f, marco_f


# return class labels from the ground truth and predictions
def get_class_labels(y_gold, y_prediction):
    return np.unique(np.concatenate((y_gold, y_prediction)))


# a functions that prints the evaluation metrics
def evaluation(y_gold, y_prediction):
    class_labels = get_class_labels(y_gold, y_prediction)

    # Confusion Matrix
    confusion = confusion_matrix(y_gold, y_prediction, class_labels)
    print(f'Confusion matrix: \n{confusion}')

    # accuracy
    print(f'Accuracy: \n{accuracy_from_confusion(confusion)}')

    # Precision
    (p_random, macro_p_random) = precision(y_gold, y_prediction, class_labels)
    print(f'Precision: \n{p_random}')
    print(f'Macro-average precision: \n{macro_p_random}')

    # Recall
    (r_random, macro_r_random) = recall(y_gold, y_prediction, class_labels)
    print(f'Recall: \n{r_random}')
    print(f'Macro-average recall: \n{macro_r_random}')

    # F1 score
    (f1_random, macro_f1_random) = f1_score(y_gold, y_prediction, class_labels)
    print(f'F1 scores: \n{f1_random}')
    print(f'Marco-averaged F1 scores: \n{macro_f1_random}')
    print("\n")


# shuffle the dataset and return N amount of shuffled datasets
def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices


# return n-fold of train sets and test sets
def train_test_k_fold(n_folds, n_instances):
    split_indices = k_fold_split(n_folds, n_instances)
    folds = []
    for k in range(n_folds):
        test_indices = split_indices[k]
        train_indices = np.hstack(split_indices[:k] + split_indices[k + 1:])
        folds.append([train_indices, test_indices])

    return folds


# perform the cross validations and return the best model
def cross_validation(n_folds, n_instances, x, y):
    accuracies = np.zeros((n_folds,))
    # to store the datasets to train the best decision tree model
    best_classifier = None
    classifiers_list_n_folds = []

    for i, (train_indices, test_indices) in \
            enumerate(train_test_k_fold(n_folds, n_instances)):
        x_train = x[train_indices, :]
        y_train = y[train_indices]
        x_test = x[test_indices, :]
        y_test = y[test_indices]

        # Train the decision tree model
        classifier = DecisionTreeClassifier()
        classifier.fit(x_train, y_train)
        classifiers_list_n_folds.append(classifier)

        predictions = classifier.predict(x_test)

        acc = get_accuracy(y_test, predictions)
        accuracies[i] = acc

        if acc >= max(accuracies):
            best_classifier = classifier

    return best_classifier, classifiers_list_n_folds


# take a list of classifiers and test dataset and return the major vote of the predictions
# act like a random forest model
def get_major_vote_from_random_forest(classifiers_list, x_test):
    predictions_list = []
    for i, classifier in enumerate(classifiers_list):
        prediction = classifier.predict(x_test)
        predictions_list.append(prediction)

    predictions_rf = [Counter(col).most_common(1)[0][0]
                      for col in zip(*predictions_list)]

    return predictions_rf
