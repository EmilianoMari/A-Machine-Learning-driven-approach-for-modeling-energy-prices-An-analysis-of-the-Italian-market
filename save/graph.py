import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import norm


class Graph:
    def __init__(self, market, mode="all"):
        self.market = market
        self.mode = mode

    def save(self, plot, name):
        if self.mode in ["png", "all"]:
            plot.savefig("bi-data/%s/%s.png" % (self.market, name))
        if self.mode in ["eps", "all"]:
            plot.savefig("bi-data/%s/%s.eps" % (self.market, name), format='eps')

    def simple(self, array, name):
        plt.rcParams["figure.figsize"] = (14, 10)
        fig, ax = plt.subplots()
        ax.plot(array)
        self.save(plt, name)

    def multi_simple(self, multiple_array, name):
        plt.rcParams["figure.figsize"] = (14, 10)
        fig, axs = plt.subplots(len(multiple_array))
        for i in range(len(multiple_array)):
            axs[i].plot(multiple_array[i], linewidth=1)
        self.save(plt, name)

    def cluster(self, array, color, name):
        title = name.upper()
        plt.rcParams["figure.figsize"] = (14, 10)

        fig1, ax1 = plt.subplots()
        ax1.scatter(array[:, 0], array[:, 1], c=color, cmap="viridis", s=5)
        ax1.set_title(title, fontsize=25)
        self.save(plt, name)

    def histogram(self, array, name):
        title = name.upper()
        plt.rcParams["figure.figsize"] = (14, 10)

        st = np.std(array)
        mn = scipy.mean(array)

        for i in range(len(array)):
            array[i] = (array[i] - mn) / st

        fig2, ax2 = plt.subplots()
        n, bins, columns = ax2.hist(array, 30, density=True)
        y = norm.pdf(bins, 0, 1)
        ax2.plot(bins, y, '--', linewidth=5)
        ax2.set_title(title, fontsize=25)
        self.save(plt, name)

    def decomp(self, observed, trend, seas, residual, name):
        title = name.upper()
        plt.rcParams["figure.figsize"] = (14, 10)
        fig, axs = plt.subplots(4)

        axs[0].plot(observed, linewidth=1)
        axs[1].plot(trend, linewidth=1)
        axs[2].plot(seas, linewidth=1)
        axs[3].plot(residual, linewidth=1)

        axs[0].label_outer()
        axs[1].label_outer()
        axs[2].label_outer()

        axs[0].set_title(title, fontsize=20)
        self.save(plt, name)

    def fill(self, original, filled, name):
        title = name.upper()
        plt.rcParams["figure.figsize"] = (14, 10)

        f = []
        o = []

        for rv, v in zip(original, filled):
            o.append(None) if rv == "None" else o.append(rv)
            f.append(v)

        plt.title(title, fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.plot(f)
        plt.plot(o)
        plt.legend(["refilled", "original"], fontsize=20)

        self.save(plt, name)