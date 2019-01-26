# Given a data set of string protien families and another set of strings of a protien sequence

import numpy as np
import pandas as pd
import heapq


# Load data from txt file
def read_str_features(filename):
    # Open file and read all lines from file
    open_file = open(filename, 'r')
    contents = open_file.readlines()

    # Empty list fo features and labels
    features = []
    labels = []

    # Store feature string in features and label in labels
    for i in range(len(contents)):
        feature_label = contents[i].strip('\n').split(' ')
        features.append(feature_label[0])
        labels.append(int(feature_label[1]))

    # Close file
    open_file.close()

    # Convert label list to np array
    labels = np.array(labels)

    return features, labels


def count_substrings(s, p):
    counts = {}
    for i in range(len(s) - p + 1):
        substring = s[i:i + p]
        if substring not in counts:
            counts[substring] = 1
        else:
            counts[substring] += 1
    return counts


# String Kernel algorithm that implicitly computes feature map through dot product of counts
def kernel_p1(s, t, p):
    count = 0
    substrings_s = count_substrings(s, p)
    substrings_t = count_substrings(t, p)

    # Find all substrings of length p that are in common with t
    for key in substrings_s:
        if key in substrings_t:
            count += substrings_s[key] * substrings_t[key]

    return count


# String Kernel algorithm that implicitly computes feature map through dot product of counts
def kernel_p2(s, t, p):
    count = 0
    s_counts = count_substrings(s, p)
    t_counts = count_substrings(t, p)

    for key in s_counts:
        if key in t_counts:
            count += 1

    return count


def kernelized_perceptron_p1(x, y, p):

    # Initialize an empty array to store indices of mistakes
    mistakes = []

    # Append first x to mistakes since the first dot product is always mistake
    mistakes.append(0)

    for t in range(len(x) - 1):
        dot_prod = 0
        for i in range(len(mistakes)):
            dot_prod += y[mistakes[i]]*kernel_p1(x[mistakes[i]], x[t+1], p)
        if (y[t + 1] * dot_prod) <= 0:
            # Keep track of the index of the mistake
            mistakes.append(t+1)

    return mistakes


def kernelized_perceptron_p2(x, y, p):

    # Initialize an empty array to store indices of mistakes
    mistakes = []

    # Append first x to mistakes since the first dot product is always mistake
    mistakes.append(0)

    for t in range(len(x) - 1):
        dot_prod = 0
        for i in range(len(mistakes)):
            dot_prod += y[mistakes[i]]*kernel_p2(x[mistakes[i]], x[t+1], p)
        if (y[t + 1] * dot_prod) <= 0:
            # Keep track of the index of the mistake
            mistakes.append(t+1)

    return mistakes


def perceptron_error(x, y, mistakes, p):

    correct_test = len(x)

    # Error of ith pass of perceptron
    for t in range(len(x) - 1):
        dot_prod = 0
        for i in range(len(mistakes)):
            dot_prod += y[mistakes[i]] * kernel_p1(x[mistakes[i]], x[t + 1], p)
        if (y[t + 1] * dot_prod) <= 0:
            correct_test = correct_test - 1

    return 1 - float(correct_test) / float(len(x))


def perceptron_test_error(t_features, t_labels, x, y, mistakes, p):

    correct_test = len(t_features)

    # Test error of ith pass of perceptron
    for t in range(len(t_features)):
        dot_prod = 0
        for i in range(len(mistakes)):
            dot_prod += y[mistakes[i]] * kernel_p1(x[mistakes[i]], t_features[t], p)
        if (t_labels[t] * dot_prod) <= 0:
            correct_test = correct_test - 1

    return 1 - float(correct_test) / float(len(t_features))


def main():
    train_features, train_labels = read_str_features('pa4train.txt')
    test_features, test_labels = read_str_features('pa4test.txt')

    # Part 1
    print('Part 1:')

    classifier1 = kernelized_perceptron_p1(train_features, train_labels, 3)

    classifier_error1 = perceptron_error(train_features, train_labels, classifier1, 3)

    print('Training error for kernelized perceptron with p = 3 is: ' + str(classifier_error1))

    classifier2 = kernelized_perceptron_p1(train_features, train_labels, 4)

    classifier_error2 = perceptron_error(train_features, train_labels, classifier2, 4)

    print('Training error for kernelized perceptron with p = 4 is: ' + str(classifier_error2))

    classifier3 = kernelized_perceptron_p1(train_features, train_labels, 5)

    classifier_error3 = perceptron_error(train_features, train_labels, classifier3, 5)

    print('Training error for kernelized perceptron with p = 5 is: ' + str(classifier_error3) + '\n')

    # Test Error

    test_error1 = perceptron_test_error(test_features, test_labels, train_features, train_labels, classifier1, 3)

    test_error2 = perceptron_test_error(test_features, test_labels, train_features, train_labels, classifier2, 4)

    test_error3 = perceptron_test_error(test_features, test_labels, train_features, train_labels, classifier3, 5)

    print('Test error for kernelized perceptron with p = 3 is: ' + str(test_error1))

    print('Test error for kernelized perceptron with p = 4 is: ' + str(test_error2))

    print('Test error for kernelized perceptron with p = 5 is: ' + str(test_error3) + '\n')

    d = {'# of passes': [3, 4, 5], 'Training error': [classifier_error1, classifier_error2, classifier_error3],
         'Test Error': [test_error1, test_error2, test_error3]}

    data_frame = pd.DataFrame(data=d)

    print(data_frame)
    print('\n')


    # Part 2
    print('Part 2:')

    classifier1 = kernelized_perceptron_p2(train_features, train_labels, 3)

    classifier_error1 = perceptron_error(train_features, train_labels, classifier1, 3)

    print('Training error for kernelized perceptron with p = 3 is: ' + str(classifier_error1))

    classifier2 = kernelized_perceptron_p2(train_features, train_labels, 4)

    classifier_error2 = perceptron_error(train_features, train_labels, classifier2, 4)

    print('Training error for kernelized perceptron with p = 4 is: ' + str(classifier_error2))

    classifier3 = kernelized_perceptron_p2(train_features, train_labels, 5)

    classifier_error3 = perceptron_error(train_features, train_labels, classifier3, 5)

    print('Training error for kernelized perceptron with p = 5 is: ' + str(classifier_error3) + '\n')


    # Test Error
    test_error1 = perceptron_test_error(test_features, test_labels, train_features, train_labels, classifier1, 3)

    test_error2 = perceptron_test_error(test_features, test_labels, train_features, train_labels, classifier2, 4)

    test_error3 = perceptron_test_error(test_features, test_labels, train_features, train_labels, classifier3, 5)

    print('Test error for kernelized perceptron with p = 3 is: ' + str(test_error1))

    print('Test error for kernelized perceptron with p = 4 is: ' + str(test_error2))

    print('Test error for kernelized perceptron with p = 5 is: ' + str(test_error3) + '\n')

    d = {'# of passes': [3, 4, 5], 'Training error': [classifier_error1, classifier_error2, classifier_error3],
         'Test Error':[test_error1, test_error2, test_error3]}

    data_frame = pd.DataFrame(data=d)

    print(data_frame)
    print('\n')

    # Part 5

    # get the classifier from part 2
    classifier = classifier3

    # dictionary to hold all substrings and there counts from mistakes
    dict = {}

    # Find the number of occurrences of substrings in mistakes
    for i in range(len(classifier)):
        substrings = count_substrings(train_features[i], 5)
        for key in substrings:
            if key not in dict:
                dict[key] = substrings[key]
            else:
                dict[key] += substrings[key]

    # Get the largest 2
    results = heapq.nlargest(2, dict, key=dict.get)

    # Print the largest 2 results
    print(results)
    return 0


if __name__ == "__main__":
    main()