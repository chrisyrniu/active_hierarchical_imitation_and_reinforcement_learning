import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import seaborn as sns

import pickle

fig = plt.figure()
def plot():
    dct = {"HIL_RL_active":('r','-',"noise based"),"HIL_RL_nn_bag":('g','--',"multi policy"),"HIL_RL_dagger":('b',':',"pure dagger")}
    label_list = []
    for path in dct:
        if "dagger" not in path:
            al_flag = True
        else:
            al_flag = False
        succ = []
        ec = []
        for expnum in range(1,13):
            al = "al_" if al_flag else ""
            filename = "exp%d_%sexpert_cost.pkl" % (expnum,al)
            pkl_dir = os.path.join(path,filename)
            if not os.path.exists(pkl_dir):
                continue
            file = open(pkl_dir, "rb")
            data = pickle.load(file)
            file.close()
            succ.append(data[1])
            ec.append(data[0])
        ec = np.array(ec)
        ec = np.mean(ec,axis=0)
        label = path[7:]
        label_list.append(dct[path][2])
        sns.tsplot(time=ec,
                   data=succ, color=dct[path][0],
                   linestyle=dct[path][1],legend=True)
    plt.ylabel("Success Rate", fontsize = 10)
    plt.xlabel("Expert Cost", fontsize = 10)
    plt.legend(labels=label_list)
    plt.title("Expert Cost", fontsize = 20)
    plt.savefig("ec.png")

if __name__=="__main__":
    plot()
