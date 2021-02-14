import numpy as np


def read_file(filename):
    x = []
    y_labels = []
    for line in open(filename):
        if line.strip() != " ":
            row = line.strip().split(',')
            x.append(list(map(int, row[:-1])))
            y_labels.append(row[-1])

    [Classes, y] = np.unique(y_labels, return_inverse=True)

    x = np.array(x)
    y = np.array(y)
    return x, y, Classes


def print_distribution(y, classes):
    total = len(y)
    for i in range(len(classes)):
        amount = np.count_nonzero(y == i)
        percentage = round((amount / total) * 100, 2)
        print(f'{classes[i]}: {amount} ({percentage}%)')


if __name__ == "__main__":
    (x, y, classes) = read_file("data/toy.txt")
    print(x.shape)
    print(y.shape)
    print(classes)
    print_distribution(y, classes)
    print("\n")

    (x, y, classes) = read_file("data/train_full.txt")
    print_distribution(y, classes)
    print("\n")

    (x, y, classes) = read_file("data/train_sub.txt")
    print_distribution(y, classes)
