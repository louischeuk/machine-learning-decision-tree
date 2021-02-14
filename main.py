from classification import DecisionTreeClassifier
from evaluation import *
import classification

if __name__ == "__main__":
    print("Loading the training dataset...\n")
    x, y = classification.read_dataset("data/train_sub.txt")

    # Make predictions on the training dataset - should get 1
    print("Training the decision tree...")
    classifier = DecisionTreeClassifier()
    classifier.fit(x, y)
    predictions = classifier.predict(x)
    print("Accuracy on TRAINING\n", (predictions == y).mean(), "\n")

    # Make predictions on the test dataset
    print("Loading the test set...")
    x_test, y_test = classification.read_dataset("data/test.txt")
    print("Making predictions on the test set...")
    predictions = classifier.predict(x_test)
    print("Accuracy on TEST\n", (predictions == y_test).mean(), "\n")

    # Evaluation - confusion matrix, precision, recall, F1 scores
    n_folds = 10
    print("Evaluate the predictions of the Decision Tree models... ")
    evaluation(y_test, predictions)

    # Evaluate the model using cross validation
    print("Picking the Decision Tree model that provides highest accuracy from cross-validation...")
    print("Generating the N amount of Decision Tree models for n-folds...")

    # output: best classifier, list of classifiers
    classifier_cv, classifiers_list_n_folds = cross_validation(n_folds, len(x), x, y)
    print("and now evaluate the best model on test.txt...")
    predictions = classifier_cv.predict(x_test)
    print("Accuracy on TEST with cross validation\n", (predictions == y_test).mean(), "\n")

    # random forest
    print("Using random forest to make predictions on the test.txt...")
    predictions_rf = get_major_vote_from_random_forest(classifiers_list_n_folds, x_test)
    print("Accuracy on TEST with random forest\n", (predictions_rf == y_test).mean(), "\n")

    # Pruning - accuracy may (not) improve
    print("Loading the validation.txt...")
    x_val, y_val = classification.read_dataset("data/validation.txt")
    print("Pruning...")
    classifier.prune(x_val, y_val)
    print("Making predictions on the test.txt...")
    predictions = classifier.predict(x_test)
    print("Accuracy on TEST after pruning \n", (predictions == y_test).mean(), "\n")
