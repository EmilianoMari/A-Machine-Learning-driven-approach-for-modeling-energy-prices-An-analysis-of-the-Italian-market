import numpy as np
import pandas as pd
from .neural import Neural


class NeuralSequence(Neural):
    def fill_holes(self, df, value_scalar):
        predicted = np.array([])
        index_to_remove = []
        for i in range(len(self.metrics)):
            index_to_remove.append(i)

        array = np.array([])
        for index, row in df.iterrows():
            if pd.isnull(row[0]):
                while len(array) != len(self.metrics) * self.dataset_len:
                    array = np.append(array, "0")
                output = self.predict(array)
                predicted = np.append(predicted, output[0][0] / value_scalar)
                for metric, value_array in zip(self.metrics, output):
                    value = value_array[0] / value_scalar
                    df.loc[index, metric] = value
                    row[metric] = value

            if len(array) == len(self.metrics) * self.dataset_len:
                array = np.delete(array, index_to_remove)
            if not pd.isnull(row[0]):
                array = np.append(array, row)
        return df, predicted

    def refill(self, df, value_scalar):
        series = np.array([])
        array = np.array([])
        for index, row in df.iterrows():
            if len(array) == self.dataset_len * self.n_metrics:
                output = self.predict(array)
                series = np.append(series, output[0][0]/value_scalar)
                array = np.delete(array, [0, 1])
            else:
                series = np.append(series, row[0])
            array = np.append(array, row[0])
            array = np.append(array, row[1])
        return np.log(series)

    def prob_refill(self, array, first, length, b, min_prob, value_scalar):
        trajectory = np.array([first])
        log_ret = np.array([])
        prob_array = np.array([])
        all_probability = np.array([])

        array = np.exp(array)
        for i in range(length-1):
            output = self.predict(array)
            all_probability = np.append(all_probability, (int(output[0][1])+int(output[1][1]))/2)
            if int(output[0][1]) > min_prob and int(output[1][1]) > min_prob:
                value = np.log(output[0][0] / value_scalar)
                diff = np.log(output[1][0] / value_scalar)
                prob_array = np.append(prob_array, (int(output[0][1])+int(output[1][1]))/2)
            else:
                diff = b[1] * np.random.normal(0, 1) - b[0] * trajectory[i]
                value = trajectory[i] + diff

            log_ret = np.append(log_ret, diff)
            trajectory = np.append(trajectory, value)
            array = np.delete(array, [0, 1])
            array = np.append(array, np.exp(value))
            array = np.append(array, np.exp(diff))
        return log_ret, trajectory, prob_array, all_probability

    def recognize_pattern(self, array):
        dic = {}
        i = 0
        while i < len(array):
            if array[i] in array[i + 1:]:
                index = array[i + 1:].index(array[i])
                length = self.seq_len(array[i:], array[i + index + 1:])
                i += length
                if length not in dic:
                    dic[length] = 0
                dic[length] += 1
                continue
            i += 1
        return self.check_loop(dic)

    @staticmethod
    def seq_len(a1, a2):
        i = 0
        for e1, e2 in zip(a1, a2):
            if e1 != e2:
                break
            i += 1
        return i

    @staticmethod
    def check_loop(dic):
        for key in dic:
            if key >= 60:
                return True
        return False
