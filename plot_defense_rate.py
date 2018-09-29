import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 15
plt.rcParams['font.weight'] = 'bold'


DATA_PATH = "/Users/user/workspace/adversarial-attack/data"  # Excel files
filenames = [fname for fname in os.listdir(DATA_PATH) if '.xlsx' in fname and '~' not in fname]

for fname in filenames:
    print("Extracting information from: {}".format(fname))
    fpath = os.path.join(DATA_PATH, fname)
    df = pd.read_excel(fpath, header=None)

    model_name = df.iloc[0, 0]
    colnames = df.iloc[2:7, 0].reset_index().iloc[:, 1]
    # colnames = pd.Series([name if name != "BIM" else "PGD" for name in colnames])
    df = df.iloc[2:7, 1:6].T
    df = df.rename(columns=lambda x: colnames.values[x-2])

    markers = ['s', '8', 'X', 'P', '*']

    print(df)
    for idx, marker in zip(range(5), markers):
        plt.plot(df.iloc[:, idx], marker=marker)
    plt.xticks([1, 2, 3, 4, 5])
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    leg = plt.legend(colnames, loc='lower left')
    leg.get_frame().set_alpha(0.3)
    plt.title(model_name)
    plt.xlabel("Padding Size")
    plt.ylabel("Defense Rate")
    plt.savefig(os.path.join(DATA_PATH, 'dr_plots/{}.pdf'.format(model_name)))
    plt.clf()

    # Histogram
    # ax = plt.subplot(111)
    # ncols = len(colnames)
    # x_pos = 4 * np.arange(ncols) + 1  # [1, 5, 9, 13, 17]

    # for i in range(ncols):
        # ax.bar(x_pos-1+0.5*i, df.iloc[:, i].values, width=0.5, align='center')

    # xtick_labels = np.zeros(4*ncols).astype(int)
    # for i, x in enumerate(x_pos):
        # xtick_labels[x] = i + 1
    # plt.xticks(x_pos, range(1, 6))
    # ax.set_xlabel("Padding Size")
    # ax.set_ylabel("Defense Rate")
    # ax.legend(list(df.columns))
    # ax.set_title(model_name)

    # plt.savefig(os.path.join(DATA_PATH, 'dr_plots/{}_hist.png'.format(model_name)))
    # plt.clf()

    # Axis changed historgram
    ax = plt.subplot(111)
    ncols = len(colnames)
    x_pos = 4 * np.arange(ncols) + 1  # [1, 5, 9, 13, 17]

    for i in range(ncols):
        ax.bar(x_pos-1+0.5*i, df.iloc[i, :].values, width=0.5, align='center')

    xtick_labels = np.zeros(4*ncols).astype(int)
    for i, x in enumerate(x_pos):
        xtick_labels[x] = i + 1
    plt.xticks(x_pos, colnames)
    ax.set_ylabel("Defense Rate")
    ax.legend(range(1, 6), title="Padding Size")
    ax.set_title(model_name)

    plt.savefig(os.path.join(DATA_PATH, 'dr_plots/{}_hist.png'.format(model_name)))
    plt.clf()
