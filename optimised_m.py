import numpy as np
import matplotlib.pyplot as plt


def optimise_m(x):
    """
    :param x: int:  model evaluation performance measure(eg average accuracy from CV)
    :return:  returns optimized fuzzier for the classification label securities
    """
    m = round((1 / (x * x)) + 1)
    return round(m)


def sim_m():
    """
    :return:List : Optimised fuzzifiers based on simulated model performance measure values.
    """
    r = np.arange(0.5, 1.09, 0.1)
    m_list = [optimise_m(i) for i in r]
    return m_list, r


a, b = sim_m()
plt.plot(a, b, marker='o')
plt.xlabel('optimised m')
plt.ylabel('Test accuracy')
plt.show()
