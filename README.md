## Machine Learning: Decision Trees


### Data

The ``data/`` directory contains all the datasets.

The primary datasets are:
- ``train_full.txt``
- ``train_sub.txt``
- ``train_noisy.txt``
- ``validation.txt``

Some simpler datasets that you may use to help you with implementation or 
debugging:
- ``toy.txt``
- ``simple1.txt``
- ``simple2.txt``

The official test set is ``test.txt``. Please use this dataset sparingly and 
purely to report the results of evaluation. Do not use this to optimise your 
classifier (use ``validation.txt`` for this instead). 


### Codes

- ``classification.py``

	* Contains  ``DecisionTreeClassifier``, ``Node`` class
    * Contains helpers functions
    
    
- ``evaluation.py``

    * Contains  functions to calculate the evaluation metrics: 
        * confusion matrix
        *  precision, Marco-averaged precision
        *  recall, Marco-averaged recall
        *  F1 scores, Marco-averaged F1 scores
    * Contains functions for cross-validation
    * Contains functions for random forest
    

- ``main.py``

	* Decision tree pipeline
        * Make predictions on the training dataset - should get accuracy of 1
        * Make predictions on the test dataset
        * Evaluation metrics
        * Evaluate the model using cross validation
        * Random forest
        * Post-pruning
        

- ``read_dataset.py`` 
    * contains functions to analysis data

### Instructions

Simply running the ``main.py``.


ps. Group project


