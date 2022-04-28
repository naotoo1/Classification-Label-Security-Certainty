"""Module to Determine model classification Certainty and thresh-hold label security"""
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


class ProtoCertt:
    """
    Prototype certainty and overall model certainty
    :param
    x_test: array, shape=[num_data,num_features]
            Where num_data is the number of samples and num_features refers to the number of features.

    class_labels: array-like, shape=[num_classes]
       Class labels of prototypes

    predict_results:  array-like, shape=[num_data]
        Predicted labels of the test-set

    """

    def __init__(self, y_test, class_labels, predict_results):
        self.y_test = y_test
        self.class_labels = class_labels
        self.predict_results = predict_results

    def model_certainty_list(self, x):
        """
        Determines the list of data points whose class is the same as the class of the prototypes
        :param x:
        Test set class labels
        :return:
        List containing labels of data points whose class is the same as the class of the prototype
        """
        same_list = []
        for i in range(self.predict_results.shape[0]):
            if x[i] == self.predict_results[i]:
                same_list.append(x[i])
        return same_list

    def my_proto_cert(self, x):
        """
        Computes the model certainty of the respective prototypes
        :param x:
        Class labels of the test set
        :return:
        The model certainty with respect to each prototype
        """
        proto_cert_list = []
        d = {i: self.model_certainty_list(x).count(i) for i in self.model_certainty_list(x)}
        # tensor so u convert it to numpy list
        dict_ = {i: list(self.predict_results.numpy()).count(i) for i in list(self.predict_results.numpy())}
        # dict_ = {i: list(self.y_test).count(i) for i in list(self.y_test)}
        my_list = list(d.items())
        my_list1 = list(dict_.items())
        for label in range(self.class_labels.shape[0]):
            for i in range(len(self.class_labels)):
                if my_list[label][0] == my_list1[i][0]:
                    prototype_certainty = my_list[label][1] / my_list1[i][1]
                    proto_cert_list.append([my_list[label][0], prototype_certainty])
        return proto_cert_list

    def overall_model_cert(self, x):
        """
        Computes the overall model certainty taking into consideration all respective prototype certainties by way of
        average.
        :param x:
        Test set class labels
        :return:
        Overall model certainty
        """

        proto_cert = self.my_proto_cert(x)
        sum_ = 0
        for i in range(len(self.my_proto_cert(x))):
            sum_ += self.my_proto_cert(x)[i][1]
        sum_proto_cert = sum_ / self.class_labels.shape[0]
        return sum_proto_cert

    def thresh_function(self, x, y, y_, y__, l3):
        """
        sorting function for class related threshold in multiple reject classification
        :param x: predicted labels with their corresponding securities
        :param y: list: security threshold
        :param y_: string: '>', '<' ,'>=' to indicate the threshold security
        :param y__: string: 's' for securities , 'i' for index of data point, 'l' for label, 'all' for list with
                (securities,indexes,labels)
        :param y___: class label under consideration (None for all labels)
        :param l3: list of class labels
        :return: List containing securities( greater than or less than a given security thresh-hold),
        """
        empty = []
        empty2 = []
        empty3 = []
        empty4 = []
        # l3 = [0, 1, 2]
        for i in range(len(x)):
            for j3 in range(len(l3)):
                if y_ == '>' and x[i][0] == l3[j3] and x[i][1] > y[j3]:
                    empty.append(x[i][1])
                    empty2.append(i)
                    empty3.append(x[i][0])
                    empty4.append([i, x[i][0], x[i][1]])
                if y_ == '>=' and x[i][0] == l3[j3] and x[i][1] >= y[j3]:
                    empty.append(x[i][1])
                    empty2.append(i)
                    empty3.append(x[i][0])
                    empty4.append([i, x[i][0], x[i][1]])
                if y_ == '=' and x[i][0] == l3[j3] and x[i][1] == y[j3]:
                    empty.append(x[i][1])
                    empty3.append(x[i][0])
                    empty2.append(i)
                    empty4.append([i, x[i][0], x[i][1]])
                if y_ == '<' and x[i][0] == l3[j3] and x[i][1] < y[j3]:
                    empty.append(x[i][1])
                    empty2.append(i)
                    empty3.append(x[i][0])
                    empty4.append([i, x[i][0], x[i][1]])
        if y__ == 'i':
            return empty2
        if y__ == 's':
            return empty
        if y__ == 'l':
            return empty3
        if y__ == 'all':
            return empty4

    def thresh_y_test(self, x):
        """

        :param x: thresh hold index list
        :return: labels with the thresh hold security
        """
        empty = []
        y = self.y_test
        for index in x:
            for i in range(len(y)):
                if i == index:
                    empty.append(y[i])
        return empty


if __name__ == '__main__':
    print('import module to use')
