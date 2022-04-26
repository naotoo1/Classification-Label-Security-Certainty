import numpy as np
import scipy.interpolate
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap


# matplotlib.style.use('default')


class Contourrn:
    def __init__(self):
        pass

    def plot_dec_boundary(self, x, y, model, model_p, title, xlabel, ylabel, model_type, model_index):
        """ plot learned prototypes, with the data set as well as the decision boundary

        :param x: data set
        :param y: labels of the data set
        :param model: model under consideration
        :param model_p: array-like: model prototypes
        :param title: string: Title of graph
        :param xlabel: string: title of the x-axis
        :param ylabel: string: Title of the y-axis
        :param model_type:string: Name of the model
        :param model_index: int: model index
        :return: plot
        """

        colors = ["r", "b", "g", "y", "m"]
        colors_ = ["r", "b", "g"]

        marker = ["*", "P", "D", "p", "H"]
        cm = ListedColormap(colors_)
        ax = plt.gca()
        z1 = model_p
        # Plotting decision regions
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        x1, y1 = np.meshgrid(np.arange(x_min, x_max, 0.05),
                             np.arange(y_min, y_max, 0.05))

        y_pred_1 = model.predict(torch.Tensor(np.c_[x1.ravel(), y1.ravel()]))
        Z1 = y_pred_1.reshape(x1.shape)
        plt.contourf(x1, y1, Z1, alpha=0.4, cmap=cm)

        # plotting data points

        for t in range(len(y)):
            if y[t] == 0:
                s1 = ax.scatter(x[t, 0], x[t, 1], c='r', marker='v',
                                )
            if y[t] == 1:
                s2 = ax.scatter(x[t, 0], x[t, 1], c='b', marker='v',
                                )
            if y[t] == 2:
                s3 = ax.scatter(x[t, 0], x[t, 1], c='g', marker='v',
                                )
        legend1 = plt.legend((s1, s2, s3), ["Setosa", "Versicolor", "Virginica"], title="Iris Classes",
                             loc="upper left", fancybox=True, framealpha=0.5)
        ax.add_artist(legend1)

        # plotting the prototypes
        t1 = ax.scatter(z1[0][0], z1[0][1], s=100, color=colors[0], marker=marker[model_index])
        t2 = ax.scatter(z1[1][0], z1[1][1], s=100, color=colors[1], marker=marker[model_index])
        t3 = ax.scatter(z1[2][0], z1[2][1], s=100, color=colors[2], marker=marker[model_index])
        legend2 = plt.legend((t1, t2, t3), ["Setosa ", "Versicolor ",
                                            "Virginica"], title=f"{model_type} Prototypes", loc="lower left",
                             fancybox=True, framealpha=0.5)
        ax.add_artist(legend2)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        return plt.show()

    def plot__newt(self, x, y, label_sec, model_p, index_list, xlabel, ylabel, title, model_1, model_type, model_index,
                   h):
        """ plot classification label security of test set with indication of rejected and non rejected classification.


        :param x: X_test
        :param y: Y_test
        :param label_sec: List containing label securities
        :param model_p: model prototypes
        :param index_list: List containing(index of data point, label, label security)
        :param title: Title of Plot
        :param ylabel: Title of data dimension 2
        :param xlabel: Title of data dimension 1
        :param model_1: model understudy
        :param model_type:string : model name
        :param model_index: int : 0 for glvq, 1 for gmlvq and 2 for celvq
        :param h: int: thresh-hold classification label security
        :return: Plot
        """

        ax = plt.gca()
        k = []
        k1 = []
        colors = ["r", "b", "g", "y", "m"]
        marker = ["*", "P", "D", "p", "H"]
        z_ = label_sec
        z1 = model_p
        for j in index_list:
            k.append(x[j[0], 0])
            k1.append(x[j[0], 1])
        x1, y1 = np.linspace(np.min(k), np.max(k), len(k)), np.linspace(np.min(k1), np.max(k1), len(k1))
        x1, y1 = np.meshgrid(x1, y1)

        rbf = scipy.interpolate.Rbf(k, k1, z_, function='linear')
        zi = rbf(x1, y1)
        x_min, x_max = np.min(k), np.max(k)
        y_min, y_max = np.min(k1), np.max(k1)
        x11, y11 = np.meshgrid(np.arange(x_min, x_max, 0.05),
                               np.arange(y_min, y_max, 0.05))
        y_pred_1 = model_1.predict(torch.Tensor(np.c_[x11.ravel(), y11.ravel()]))

        Z1 = y_pred_1.reshape(x11.shape)

        plt.contour(x11, y11, Z1, levels=3,
                    colors=np.array([colors[0], colors[1], colors[1], colors[2]]))

        # plot the label securities regions
        plt.imshow(zi, vmin=np.min(z_), vmax=np.max(z_), origin='lower',
                   extent=[np.min(k), np.max(k), np.min(k1), np.max(k1)])

        # plot the predicted labels from the test set  and indicate rejected classification
        j = -1
        for j1 in index_list:
            j += 1
            if y[j] == 0 and j1[1] == 0 and j1[2] >= h:
                s1 = ax.scatter(x[j, 0], x[j, 1], color='r', marker='v')
            if y[j] == 0 and j1[1] != 0 and j1[2] >= h:
                s1_ = ax.scatter(x[j, 0], x[j, 1], color='r', marker='v')
            if y[j] == 0 and j1[1] == 0 and j1[2] < h:
                s1__ = ax.scatter(x[j, 0], x[j, 1], color='r', marker='v', edgecolor='k')
            if y[j] == 0 and j1[1] != 0 and j1[2] < h:
                s1_ = ax.scatter(x[j, 0], x[j, 1], color='r', marker='v', edgecolor='k')

            if y[j] == 1 and j1[1] == 1 and j1[2] >= h:
                s2 = ax.scatter(x[j, 0], x[j, 1], color='b', marker='v')
            if y[j] == 1 and j1[1] != 1 and j1[2] >= h:
                s2_ = ax.scatter(x[j, 0], x[j, 1], color='b', marker='v')
            if y[j] == 1 and j1[1] == 1 and j1[2] < h:
                s2__ = ax.scatter(x[j, 0], x[j, 1], color='b', marker='v', edgecolor='k')
            if y[j] == 1 and j1[1] != 1 and j1[2] < h:
                s2_ = ax.scatter(x[j, 0], x[j, 1], color='b', marker='v', edgecolor='k')

            if y[j] == 2 and j1[1] == 2 and j1[2] >= h:
                s3 = ax.scatter(x[j, 0], x[j, 1], color='g', marker='v')
            if y[j] == 2 and j1[1] != 2 and j1[2] >= h:
                s3_ = ax.scatter(x[j, 0], x[j, 1], color='g', marker='v')
            if y[j] == 2 and j1[1] == 2 and j1[2] < h:
                s3__ = ax.scatter(x[j, 0], x[j, 1], color='g', marker='v', edgecolor='k')
            if y[j] == 2 and j1[1] != 2 and j1[2] < h:
                s3_ = ax.scatter(x[j, 0], x[j, 1], color='g', marker='v', edgecolor='k')

        legend1 = plt.legend((s1, s2, s3,), ["Setosa", "Versicolor", "Virginica"], title="Iris Classes",
                             loc="upper left", bbox_to_anchor=(-0.6, 1))
        ax.add_artist(legend1)

        # plot the learned prototypes
        t1 = ax.scatter(z1[0][0], z1[0][1], s=100, color=colors[0], marker=marker[model_index])
        t2 = ax.scatter(z1[1][0], z1[1][1], s=100, color=colors[1], marker=marker[model_index])
        t3 = ax.scatter(z1[2][0], z1[2][1], s=100, color=colors[2], marker=marker[model_index])
        legend2 = plt.legend((t1, t2, t3,), ["Setosa ", "Versicolor ",
                                             "Virginica"], title=f"{model_type} Prototypes", loc="lower left",
                             bbox_to_anchor=(-0.6, 0.0))
        ax.add_artist(legend2)

        legend_list = []
        for class_, color in zip(["Setosa ", "Versicolor ", "Virginica"], ['r', 'b', 'g']):
            legend_list.append(Line2D([0], [0], marker='v', label=class_, ls='None', markerfacecolor=color,
                                      markeredgecolor='k'))
        legend3 = plt.legend(handles=legend_list, loc="center", bbox_to_anchor=(-0.5, 0.5),
                             title='Rejected classification')
        ax.add_artist(legend3)

        plt.colorbar()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        return plt.show()
