import random
import pandas as pd
import numpy as np
from lib import differences, get_results
from neural import NeuralSequence
from save import Graph, FileSave


'''
Energy
up: 6
down: 4 
Gas
up: 120
down: 89
Energy 
kt  5. 8522
st 0.1408
sk 0.2305
Gas
kt 29.1889
st 0.0635
sk 0.2321
'''

probability = {
    20: {
        "eu": 0,
        "ed": 0,
        "gu": 0,
        "gd": 0
    }
}

train = False
dataset_len = 20
scalar = 1000
epochs = 200
market = "ita"
batch_size = 16
n_test = 100
metrics = ["energy_price", "gas_price"]

energy_alpha = 0.395
energy_sigma = 0.120
energy_sigma_jump = 0.530
gas_alpha = 0.09
gas_sigma = 0.0290


if __name__ == '__main__':
    data_frame = pd.read_csv("bi-data/ita/data/pow_cluster.csv", names=["cluster"])
    energy_clusters = data_frame["cluster"].values
    data_frame = pd.read_csv("bi-data/ita/data/pow_prices.csv", names=["price"])
    energy_prices = data_frame["price"].values

    data_frame = pd.read_csv("bi-data/ita/data/gas_cluster.csv", names=["cluster"])
    gas_clusters = data_frame["cluster"].values
    data_frame = pd.read_csv("bi-data/ita/data/gas_prices.csv", names=["price"])
    gas_prices = data_frame["price"].values

    cauchy = []
    with open("bi-data/ita/data/cauchy_gas.dat", "r") as file:
        for line in file:
            cauchy.append(float(line))

    neural_director = NeuralSequence(market, 3, "director_bis", ["energy", "gas"], dataset_len, epochs, batch_size)
    graph = Graph(market, mode="png")
    file = FileSave(market)

    if train is True:
        x_train = np.array([])
        x_train_p = []
        y_train_e = []
        y_train_g = []
        for i in range(1824):
            x_train_p.append(energy_prices[i])
            x_train_p.append(gas_prices[i])
            if len(x_train_p) == dataset_len*len(metrics):
                x_train = np.append(x_train, x_train_p)
                '''
                0 = Up base
                1 = Down base
                2 = Up jump
                3 = Down jump
                '''
                if energy_clusters[i+1] != 0:
                    if energy_prices[i+2] - energy_prices[i+1] + energy_alpha * energy_prices[i+1] > 0:
                        y_train_e.append(0)
                    else:
                        y_train_e.append(1)
                else:
                    if energy_prices[i+2] > energy_prices[i+1]:
                        y_train_e.append(2)
                    else:
                        y_train_e.append(3)
                if gas_clusters[i+1] != 0:
                    if gas_prices[i + 2] - gas_prices[i + 1] + gas_alpha * gas_prices[i + 1] > 0:
                        y_train_g.append(0)
                    else:
                        y_train_g.append(1)
                else:
                    if gas_prices[i+2] > gas_prices[i+1]:
                        y_train_g.append(2)
                    else:
                        y_train_g.append(3)
                for k in range(len(metrics)):
                    x_train_p.pop(0)

        for array in [y_train_e, y_train_g]:
            dic = {}
            for e in array:
                if e not in dic:
                    dic[e] = 0
                dic[e] += 1
            print(dic)

        y_train_e = np.asarray(y_train_e)
        y_train_e.astype(int)
        y_train_g = np.asarray(y_train_g)
        y_train_g.astype(int)

        x_train = np.exp(x_train) * scalar
        x_train = x_train.reshape(int(len(x_train) / (dataset_len * len(metrics))), dataset_len * len(metrics))

        if neural_director.model_exist():
            neural_director.resume_train(x_train, [y_train_e, y_train_g], epochs)
        else:
            neural_director.create(x_train, [y_train_e, y_train_g])
    else:
        neural_director.load()

    ea = [[], [], [], []]
    ga = [[], [], [], []]
    ppe = []
    ppg = []
    a_jump_e = [[], []]
    a_jump_g = [[], []]
    for j in range(n_test):
        print(j)
        energy_trajectory = [energy_prices[0]]
        gas_trajectory = [gas_prices[0]]
        for i in range(1, dataset_len):
            value = energy_trajectory[i - 1] - energy_alpha * energy_trajectory[i - 1] + energy_sigma * np.random.normal(0, 1)
            energy_trajectory.append(value)
            value = gas_trajectory[i - 1] - gas_alpha * gas_trajectory[i - 1] + gas_sigma * np.random.normal(0, 1)
            gas_trajectory.append(value)

        input_array = []
        for e, g in zip(energy_trajectory, gas_trajectory):
            input_array.append(e)
            input_array.append(g)

        eu_jump = 0
        ed_jump = 0
        gu_jump = 0
        gd_jump = 0
        for i in range(dataset_len, 1826):
            res = neural_director.predict(np.exp(input_array) * scalar, dataset_len*len(metrics))

            # Energy
            e_is_jump = res[0][0]
            e_prob = res[0][1]
            if e_is_jump == 2 and e_prob > probability[dataset_len]["eu"]:
                ep = energy_trajectory[i - 1] + energy_sigma_jump * abs(np.random.normal(0, 1))
                eu_jump += 1
                ppe.append(e_prob)
            elif e_is_jump == 3 and e_prob > probability[dataset_len]["ed"]:
                ep = energy_trajectory[i - 1] - energy_sigma_jump * abs(np.random.normal(0, 1))
                ed_jump += 1
                ppe.append(e_prob)
            else:
                random_num = 100000
                if random_num == 0:
                    ep = energy_trajectory[i - 1] + energy_sigma_jump * abs(np.random.normal(0, 1))
                    eu_jump += 1
                elif random_num == 1:
                    ep = energy_trajectory[i - 1] - energy_sigma_jump * abs(np.random.normal(0, 1))
                    ed_jump += 1
                else:
                    if e_is_jump == 0 or e_is_jump == 2:
                        ep = energy_trajectory[i - 1] - energy_alpha * energy_trajectory[i - 1] + abs(energy_sigma * np.random.normal(0, 1))
                    else:
                        ep = energy_trajectory[i - 1] - energy_alpha * energy_trajectory[i - 1] - abs(energy_sigma * np.random.normal(0, 1))

            # Gas
            g_is_jump = res[1][0]
            g_prob = res[1][1]

            if g_is_jump == 2 and g_prob > probability[dataset_len]["gu"]:
                gp = gas_trajectory[i - 1] + random.choice(cauchy)
                gu_jump += 1
            elif g_is_jump == 3 and g_prob > probability[dataset_len]["gd"]:
                gp = gas_trajectory[i - 1] - random.choice(cauchy)
                gd_jump += 1
            else:
                random_num = 100000
                if random_num < 48:
                    gp = gas_trajectory[i - 1] + random.choice(cauchy)
                    gu_jump += 1
                elif 48 <= random_num < 80:
                    gp = gas_trajectory[i - 1] - random.choice(cauchy)
                    gd_jump += 1
                    ppg.append(g_prob)
                else:
                    if g_is_jump == 0 or g_is_jump == 2:
                        gp = gas_trajectory[i - 1] - gas_alpha * gas_trajectory[i - 1] + abs(gas_sigma * np.random.normal(0, 1))
                    else:
                        gp = gas_trajectory[i - 1] - gas_alpha * gas_trajectory[i - 1] - abs(gas_sigma * np.random.normal(0, 1))

            energy_trajectory.append(ep)
            gas_trajectory.append(gp)
            input_array.append(ep)
            input_array.append(gp)
            for k in range(len(metrics)):
                input_array.pop(0)

        a_jump_e[0].append(eu_jump)
        a_jump_e[1].append(ed_jump)
        a_jump_g[0].append(gu_jump)
        a_jump_g[1].append(gd_jump)

        # graph.multi_simple([energy_trajectory, gas_trajectory], "graph_%s" % j)
        # graph.simple(energy_trajectory, "energy/%s" % j)
        # file.save(energy_trajectory, "energy/%s.dat" % j)
        # graph.simple(gas_trajectory, "gas/%s" % j)
        # file.save(gas_trajectory, "gas/%s.dat" % j)

        kt, st, sk, mn = get_results(differences(energy_trajectory), False)
        ea[0].append(kt)
        ea[1].append(st)
        ea[2].append(sk)
        ea[3].append(mn)

        kt, st, sk, mn = get_results(differences(gas_trajectory), False)
        ga[0].append(kt)
        ga[1].append(st)
        ga[2].append(sk)
        ga[3].append(mn)

    print("Energy up jump: %s" % np.mean(a_jump_e[0]))
    print("Energy down jump: %s" % np.mean(a_jump_e[1]))
    print("Gas up jump: %s" % np.mean(a_jump_g[0]))
    print("Gas down jump: %s" % np.mean(a_jump_g[1]))
    print("Energy prob: %s" % np.mean(ppe))
    print("Gas prob: %s" % np.mean(ppg))

    print("ENERGY")
    for i in range(len(ea)):
        _, st, _, mn = get_results(ea[i], False)
        print(mn, st)
    print("GAS")
    for i in range(len(ea)):
        _, st, _, mn = get_results(ga[i], False)
        print(mn, st)
    print("=" * 20)

    print("ENERGY")
    res = [5.8522, 0.1408, 0.2305]
    for i in range(len(ea)-1):
        _, st, _, mn = get_results(ea[i], False)
        print(res[i] - mn)
    print("GAS")
    res = [29.1889, 0.0635, 0.2321]
    for i in range(len(ea)-1):
        _, st, _, mn = get_results(ga[i], False)
        print(res[i] - mn)
