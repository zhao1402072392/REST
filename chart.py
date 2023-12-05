import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter
import re
# import locals

def get_acc(file):
    df = pd.read_csv(file)
    # avg_acc = df.iloc[-1]["avg_acc"]
    # avg_acc_group0 = df.iloc[-1]["avg_acc_group:0"]
    # avg_acc_group1 = df.iloc[-1]["avg_acc_group:1"]
    # avg_acc_group2 = df.iloc[-1]["avg_acc_group:2"]
    # avg_acc_group3 = df.iloc[-1]["avg_acc_group:3"]
    avg_acc = df["avg_acc"]
    avg_acc_group0 = df["avg_acc_group:0"]
    avg_acc_group1 = df["avg_acc_group:1"]
    avg_acc_group2 = df["avg_acc_group:2"]
    avg_acc_group3 = df["avg_acc_group:3"]

    avg_acc_list = avg_acc.values.tolist()
    avg_acc_group0_list = avg_acc_group0.values.tolist()
    avg_acc_group1_list = avg_acc_group1.values.tolist()
    avg_acc_group2_list = avg_acc_group2.values.tolist()
    avg_acc_group3_list = avg_acc_group3.values.tolist()
    avg_error_list = []
    avg_error_group0_list = []
    avg_error_group1_list = []
    avg_error_group2_list = []
    avg_error_group3_list = []
    
    for acc in avg_acc_list:
        error = 1-acc
        avg_error_list.append(error)
    for acc in avg_acc_group0_list:
        error = 1-acc
        avg_error_group0_list.append(error)
    for acc in avg_acc_group1_list:
        error = 1-acc
        avg_error_group1_list.append(error)
    for acc in avg_acc_group2_list:
        error = 1-acc
        avg_error_group2_list.append(error)
    for acc in avg_acc_group3_list:
        error = 1-acc
        avg_error_group3_list.append(error)


    avg_error_AVG = np.mean(avg_error_list)
    avg_error_group0_AVG = np.mean(avg_error_group0_list)
    avg_error_group1_AVG = np.mean(avg_error_group1_list)
    avg_error_group2_AVG = np.mean(avg_error_group2_list)
    avg_error_group3_AVG = np.mean(avg_error_group3_list)

    return avg_error_AVG, avg_error_group0_AVG, avg_error_group1_AVG, avg_error_group2_AVG, avg_error_group3_AVG

def get_all_csv(path):
    csv_list = os.listdir(path)
    random_train_list = []
    random_test_list = []
    gradient_train_list = []
    gradient_test_list = []
    for name in csv_list:
        name_split = name.split('_')
        spare = name_split[0]
        density = name_split[1]
        seed = name_split[2]
        train = name_split[3]
        # print(name_split)
        if spare == 'random' and train == 'train.csv':
            random_train_list.append(name)
        if spare == 'random' and train == 'test.csv':
            random_test_list.append(name)
        if spare == 'gradient' and train == 'train.csv':
            gradient_train.append(name)
        if spare == 'gradient' and train == 'test.csv':
            gradient_test_list.append(name)
    # random_train = sorted_nicely(random_train)
    # random_test = sorted_nicely(random_test)
    # gradient_train = sorted_nicely(gradient_train)
    # gradient_train = sorted_nicely(gradient_train)
    print("random_train_list:", random_train_list)
    return random_train_list, random_test_list, gradient_train_list, gradient_test_list


# def sorted_nicely(l):
#     """ Sort the given iterable in the way that humans expect."""
#     convert = lambda text: int(text) if text.isdigit() else text
#     alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
#     return sorted(l, key=alphanum_key)

def get_acc_list(file_list):

    seed_list = []
    for file in file_list:
        name_split = file.split('_')
        seed = name_split[2]
        seed_list.append(seed)

    if len(set(seed_list)) == 1:
        file_list_3seed = []
        # name = locals()
        for file in file_list:
            file_list_3seed.append(file)
            file_list_3seed.append(file)
            file_list_3seed.append(file)
        #     name_split = file.split('_')
        #     file_list_3seed.append(file)
        #     for seed23 in [39, 58]:
        #         name[name_split[0] + '_' + name_split[1] + '_' + str(seed23) + '_' + name_split[3]] = file
        #         file_list_3seed.append(name_split[0] + '_' + name_split[1] + '_' + str(seed23) + '_' + name_split[3])
    else:
        file_list_3seed = file_list

    n = len(file_list_3seed)
    list_17 = file_list_3seed[0:n:3]
    list_39 = file_list_3seed[1:n:3]
    list_58 = file_list_3seed[2:n:3]

    dense_list = None
    print("list_17", list_17)
    return list_17, list_39, list_58


def get_acc_list_of_different_seed(dir, random_train_list):
    list_17, list_39, list_58 = get_acc_list(random_train_list)
    all_seed_avg_acc_list = []
    all_seed_avg_acc_group0_list = []
    all_seed_avg_acc_group1_list = []
    all_seed_avg_acc_group2_list = []
    all_seed_avg_acc_group3_list = []

    for list in list_17, list_39, list_58:
        avg_acc_list = []
        avg_acc_group0_list = []
        avg_acc_group1_list = []
        avg_acc_group2_list = []
        avg_acc_group3_list = []
        for diff_density in list:
            path = dir + diff_density
            avg_acc, avg_acc_group0, avg_acc_group1, avg_acc_group2, avg_acc_group3 = get_acc(path)
            avg_acc_list.append(avg_acc)
            avg_acc_group0_list.append(avg_acc_group0)
            avg_acc_group1_list.append(avg_acc_group1)
            avg_acc_group2_list.append(avg_acc_group2)
            avg_acc_group3_list.append(avg_acc_group3)
            print('avg_acc_list',avg_acc_group0_list)
        all_seed_avg_acc_list.append(avg_acc_list)
        all_seed_avg_acc_group0_list.append(avg_acc_group0_list)
        all_seed_avg_acc_group1_list.append(avg_acc_group1_list)
        all_seed_avg_acc_group2_list.append(avg_acc_group2_list)
        all_seed_avg_acc_group3_list.append(avg_acc_group3_list)
    print('all_seed_avg_acc_group3_list', all_seed_avg_acc_group3_list)
    return all_seed_avg_acc_list, all_seed_avg_acc_group0_list, all_seed_avg_acc_group1_list, all_seed_avg_acc_group2_list, all_seed_avg_acc_group3_list


def mean_list(loss1_list, loss2_list, loss3_list):
    mean_values = []
    std_vallues = []
    # import ipdb
    # ipdb.set_trace()
    for i, loss_tuple in enumerate(zip(loss1_list, loss2_list, loss3_list)):
        loss_list = list(loss_tuple)

        mean = np.mean(loss_list)
        std = np.std(loss_list)
        mean_values.append(mean)
        std_vallues.append(std)
    # print(len(mean_values))
    # print(len(std_vallues))
    return mean_values, std_vallues


def plot_lines(loss1, loss2, loss3, color, linestyle=None, label=None):
    # print(loss1)
    ap_mean_values, ap_std_vallues = mean_list(loss1, loss2, loss3)
    print(len(ap_mean_values))
    id = list(range(0, len(ap_mean_values)))
    ap_std_down = [ap_mean_values[x] - ap_std_vallues[x] for x in range(len(ap_mean_values))]
    ap_std_up = [ap_mean_values[x] + ap_std_vallues[x] for x in range(len(ap_mean_values))]
    l1 = ax1.plot(id, ap_mean_values, color=color, linestyle=linestyle, label=label)
    ax1.fill_between(id, ap_std_down, ap_std_up, color=color, alpha=0.3)


def put_aac_to_plot_line(dir, sparse_train, sparse_test):
    train_all_seed_avg_acc_list, train_all_seed_avg_acc_group0_list, train_all_seed_avg_acc_group1_list, \
    train_all_seed_avg_acc_group2_list, train_all_seed_avg_acc_group3_list = get_acc_list_of_different_seed(dir,
        sparse_train)

    test_all_seed_avg_acc_list, test_all_seed_avg_acc_group0_list, test_all_seed_avg_acc_group1_list, \
    test_all_seed_avg_acc_group2_list, test_all_seed_avg_acc_group3_list = get_acc_list_of_different_seed(dir, sparse_test)

    train_avg_acc1, train_avg_acc2, train_avg_acc3 = train_all_seed_avg_acc_list
    train_avg_acc1 = train_avg_acc1[:-1]
    train_avg_acc2 = train_avg_acc2[:-1]
    train_avg_acc3 = train_avg_acc3[:-1]
    train_avg_acc1_dense = train_avg_acc1[-1] * len(train_avg_acc1)
    train_avg_acc2_dense = train_avg_acc2[-1] * len(train_avg_acc1)
    train_avg_acc3_dense = train_avg_acc3[-1] * len(train_avg_acc1)
    test_avg_acc1, test_avg_acc2, test_avg_acc3 = test_all_seed_avg_acc_list
    print("test_avg_acc1_dense", len(test_avg_acc1))
    test_avg_acc1 = test_avg_acc1[:-1]
    test_avg_acc2 = test_avg_acc2[:-1]
    test_avg_acc3 = test_avg_acc3[:-1]
    test_avg_acc1_dense = [test_avg_acc1[-1]] * len(test_avg_acc1)
    test_avg_acc2_dense = [test_avg_acc2[-1]] * len(test_avg_acc1)
    test_avg_acc3_dense = [test_avg_acc3[-1]] * len(test_avg_acc1)


    train_worst_acc1, train_worst_acc2, train_worst_acc3 = train_all_seed_avg_acc_group3_list
    train_worst_acc1 = train_worst_acc1[:-1]
    train_worst_acc2 = train_worst_acc2[:-1]
    train_worst_acc3 = train_worst_acc3[:-1]
    train_worst_acc1_dense = [train_worst_acc1[-1]] * len(train_worst_acc1)
    train_worst_acc2_dense = [train_worst_acc2[-1]] * len(train_worst_acc1)
    train_worst_acc3_dense = [train_worst_acc3[-1]] * len(train_worst_acc1)

    test_worst_acc1, test_worst_acc2, test_worst_acc3 = test_all_seed_avg_acc_group3_list
    test_worst_acc1 = test_worst_acc1[:-1]
    test_worst_acc2 = test_worst_acc2[:-1]
    test_worst_acc3 = test_worst_acc3[:-1]
    test_worst_acc1_dense = [test_worst_acc1[-1]] * len(test_worst_acc1)
    test_worst_acc2_dense = [test_worst_acc2[-1]] * len(test_worst_acc1)
    test_worst_acc3_dense = [test_worst_acc3[-1]] * len(test_worst_acc1)


    return train_avg_acc1, train_avg_acc2, train_avg_acc3, \
           test_avg_acc1, test_avg_acc2, test_avg_acc3, \
           train_worst_acc1, train_worst_acc2, train_worst_acc3, \
           test_worst_acc1, test_worst_acc2, test_worst_acc3, \
           train_avg_acc1_dense, train_avg_acc2_dense, train_avg_acc3_dense, \
           test_avg_acc1_dense, test_avg_acc2_dense, test_avg_acc3_dense, \
           train_worst_acc1_dense, train_worst_acc2_dense, train_worst_acc3_dense, \
           test_worst_acc1_dense, test_worst_acc2_dense, test_worst_acc3_dense



# result_csv = "./logs1/ResNet18_0.1_gradient_17_test.csv"
all_csv_path = './logs_yesre_CelebA/'
random_train, random_test, gradient_train, gradient_test = get_all_csv(all_csv_path)
train_list_17, train_list_39, train_list_58 = get_acc_list(random_train)
test_list_17, test_list_39, test_list_58 = get_acc_list(random_test)
train_dense_avg_acc_list = []
train_dense_worst_acc_list = []
test_dense_avg_acc_list = []
test_dense_worst_acc_list = []
# for list_ in train_dense_list:
#     path = './logs_bird/' + list_
#     train_avg_acc, train_avg_acc_group0, train_avg_acc_group1, train_avg_acc_group2, train_avg_acc_group3 = get_acc(
#         path)
#     train_dense_avg_acc_list.append(train_avg_acc)
#     train_dense_worst_acc_list.append(train_avg_acc_group3)
# for list_ in test_dense_list:
#     path = './logs_bird/' + list_
#     test_avg_acc, test_avg_acc_group0, test_avg_acc_group1, test_avg_acc_group2, test_avg_acc_group3 = get_acc(path)
#     test_dense_avg_acc_list.append(test_avg_acc)
#     test_dense_worst_acc_list.append(test_avg_acc_group3)

sparse = 'random'
if sparse == 'random':
    train_avg_acc1, train_avg_acc2, train_avg_acc3, \
    test_avg_acc1, test_avg_acc2, test_avg_acc3, \
    train_worst_acc1, train_worst_acc2, train_worst_acc3, \
    test_worst_acc1, test_worst_acc2, test_worst_acc3, \
    train_avg_acc1_dense, train_avg_acc2_dense, train_avg_acc3_dense, \
    test_avg_acc1_dense, test_avg_acc2_dense, test_avg_acc3_dense, \
    train_worst_acc1_dense, train_worst_acc2_dense, train_worst_acc3_dense, \
    test_worst_acc1_dense, test_worst_acc2_dense, test_worst_acc3_dense = put_aac_to_plot_line(all_csv_path, random_train, random_test)
elif sparse == 'gradient':
    train_avg_acc1, train_avg_acc2, train_avg_acc3, \
    test_avg_acc1, test_avg_acc2, test_avg_acc3, \
    train_worst_acc1, train_worst_acc2, train_worst_acc3, \
    test_worst_acc1, test_worst_acc2, test_worst_acc3, \
    train_avg_acc1_dense, train_avg_acc2_dense, train_avg_acc3_dense, \
    test_avg_acc1_dense, test_avg_acc2_dense, test_avg_acc3_dense, \
    train_worst_acc1_dense, train_worst_acc2_dense, train_worst_acc3_dense, \
    test_worst_acc1_dense, test_worst_acc2_dense, test_worst_acc3_dense = put_aac_to_plot_line(all_csv_path, gradient_train, gradient_test)
# 设置全局格式，包括字体风格和大小等等
# 这里主要用来修改文本字体里面的格式
font_size = 20
config = {
    "font.family": 'serif',
    "font.size": font_size,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


# 修改x轴的显示方式，科学计数法
def formatnumx(x, pos):
    return '%d' % (x / 1000)


formatterx = FuncFormatter(formatnumx)
fig, ax1 = plt.subplots(figsize=(8, 8), dpi=100)
fig.legend(fontsize=font_size)
# print(train_avg_acc1)
ap = plot_lines(train_avg_acc1, train_avg_acc2, train_avg_acc3, 'red', label='train_avg')
ap = plot_lines(test_avg_acc1, test_avg_acc2, test_avg_acc3, 'lightcoral', label='test_avg')
ap = plot_lines(train_worst_acc1, train_worst_acc2, train_worst_acc3, 'dodgerblue','--', label='train_worst')
ap = plot_lines(test_worst_acc1, test_worst_acc2, test_worst_acc3, 'lightskyblue', '--', label='test_worst')

ap = plot_lines(test_avg_acc1_dense, test_avg_acc2_dense, test_avg_acc3_dense, 'goldenrod', label='test_avg_dense')
ap = plot_lines(test_worst_acc1_dense, test_worst_acc2_dense, test_worst_acc3_dense, 'goldenrod', '--', label='test_worst_dense')

ax1.set_xlabel(r'Yesre_random_Density', fontdict={'family': 'Times New Roman', 'size': font_size})
ax1.set_ylabel('Error', fontdict={'family': 'Times New Roman', 'size': font_size})
ax1.tick_params(labelsize=font_size)

ticks = ax1.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
yticks = ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
labels = ax1.set_xticklabels(['0.01', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'],
                             rotation=30, fontsize='small')
fig.tight_layout()
fig.legend(fontsize=10)
# fname_path = './logs/' + sparse + '.pdf'
# plt.savefig(fname_path)
plt.show()