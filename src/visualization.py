import os
import matplotlib.pyplot as plt
import numpy as np


def plot_one(data_folder, plot_dir):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    run_files = os.listdir(data_folder)
    for sr_file in run_files:
        with open(f'{data_folder}/{sr_file}', 'r') as f:
            sr_data = f.readline().strip().split(' ')
            sr_data = list(map(lambda x: float(x), sr_data))
            if len(sr_data) == 5:
                plt.plot([8, 16, 32, 64, 128], sr_data, 'o--', linewidth=1, label=sr_file.split('.')[0])
            else:
                plt.plot([10, 20, 30, 40, 50, 60], sr_data, 'o--', linewidth=1, label=sr_file.split('.')[0])
    plt.title(f'Generalizing to bigger problems \n ({data_folder.split("/")[-1]})')
    plt.xlabel('Number of iterations')
    plt.ylabel('SAT Accuracy (%)')

    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_dict = {}
    sorting = dict(zip(labels, handles))
    sorting = sorted(sorting.items(), key=lambda kv: int(kv[0].split('sr')[-1]))
    for item in sorting:
        sorted_dict[item[0]] = item[1]
    labels = sorted_dict.keys()
    handles = sorted_dict.values()
    plt.legend(handles, labels)

    plt.savefig(f'{plot_dir}/res_{data_folder.split("/")[-1]}.png', bbox_inches='tight')
    plt.clf()

def plot_all(data_folder, plot_dir):
    run_list = os.listdir(data_folder)
    iter_count_list = [8, 16, 32, 64, 128]
    for run in run_list:
        run_files = os.listdir(f'{data_folder}/{run}')
        for sr_file in run_files:
            with open(f'{data_folder}/{run}/{sr_file}', 'r') as f:
                sr_data = f.readline().strip().split(' ')
                sr_data = list(map(lambda x: float(x), sr_data))
                if len(sr_data) == 5:
                    plt.plot([8, 16, 32, 64, 128], sr_data, 'o--', linewidth=1, label=sr_file.split('.')[0])
                else:
                    plt.plot([10, 20, 30, 40, 50, 60], sr_data, 'o--', linewidth=1, label=sr_file.split('.')[0])
        plt.title(f'Generalizing to bigger problems \n ({run})')
        plt.xlabel('Number of iterations')
        plt.ylabel('Assignment Accuracy (%)')

        handles, labels = plt.gca().get_legend_handles_labels()
        sorted_dict = {}
        sorting = dict(zip(labels, handles))
        sorting = sorted(sorting.items(), key=lambda kv: int(kv[0].split('sr')[-1]))
        for item in sorting:
            sorted_dict[item[0]] = item[1]
        labels = sorted_dict.keys()
        handles = sorted_dict.values()
        plt.legend(handles, labels)

        plt.savefig(f'{plot_dir}/generalizing_neurosat_{run}.png', bbox_inches='tight')
        plt.clf()


def plot_merge(run1, run2, data_folder, plot_dir, x_axis, fontsize=14):
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    run_list = [run1, run2]
    color_list = []
    plot_config_list = []
    plt.figure(figsize=(4.5,3))
    # plt.figure(figsize=(6,3))
    for idx, run in enumerate(run_list):
        run_files = os.listdir(f'{data_folder}/{run}')
        plot_config = {
            'line': 'x--' if idx == 0 else 'o-',
            # 'label_prefix': run.split('/')[-1].split('_')[2] if 'nnsat' not in run.split('/')[-1].split('_')[2] else f'scaled-{plot_config_list[0]["label_prefix"]}',
            'label_prefix': run.split('/')[-1].split('_')[2] if 'nnsat' not in run.split('/')[-1].split('_')[2] else f"tensat-{run.split('/')[-1].split('_')[2].split('-')[-1]}"
        }
        plot_config_list.append(plot_config)
        # run_files = sorted(run_files, key=lambda x: int(x.split('.')[0].split('sr')[-1]))
        for sr_idx, sr_file in enumerate(run_files):
            # print(sr_file)
            with open(f'{data_folder}/{run}/{sr_file}', 'r') as f:
                sr_data = f.readline().strip().split(' ')
                sr_data = list(map(lambda x: float(x), sr_data))
                # print(sr_data)
                if len(color_list) != len(run_files):
                    p = plt.plot(x_axis, sr_data, plot_config['line'], linewidth=1)
                    plt.plot([], [], color=p[0].get_color(), label=f'{sr_file.split(".")[0]}')
                    color_list.append(p[0].get_color())
                else:
                    p = plt.plot(x_axis, sr_data, plot_config['line'], linewidth=1, color=color_list[sr_idx])
        plt.plot([], [], plot_config['line'], color='b', label=f'{plot_config["label_prefix"]}')


    # plt.title(f'Stable accuracy when iteration time larger than 60', fontsize=fontsize)
    # plt.title(f'Generalizing to bigger problems \n neurosat vs. tensat (lstm)', fontsize=fontsize)
    # plt.title(f'Generalizing to bigger problems \n {run1.split("_")[-3].split("-")[0]} vs. tensat ({run1.split("_")[-3].split("-")[1]})', fontsize=fontsize)
    # plt.xlabel('Number of iterations', fontsize=fontsize)
    plt.ylabel('SAT Accuracy (%)', fontsize=fontsize)
    # plt.ylabel('Assignment Accuracy (%)', fontsize=fontsize)

    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_dict = {}
    sorting = dict(zip(labels, handles))
    numerical = {}
    non_numer = {}
    for item in sorting:
        if 'sr' in item:
            numerical[item] = sorting[item]
        else:
            non_numer[item] = sorting[item]
    sorting = numerical
    sorting = sorted(sorting.items(), key=lambda kv: int(kv[0].split('sr')[-1]))
    for item in non_numer:
        sorted_dict[item] = non_numer[item]
    for item in sorting:
        sorted_dict[item[0]] = item[1]
    labels = sorted_dict.keys()
    handles = sorted_dict.values()
    # plt.legend(handles, labels, labelspacing=0.01, fontsize=fontsize, framealpha=0, bbox_to_anchor=(1.05, 1.0), loc='upper left')
    # plt.legend(handles, labels, labelspacing=0.01, framealpha=0)# bbox_to_anchor=(1.05, 1.0), loc='upper left')

    plt.savefig(f'{plot_dir}/generalizing_{"_".join([run.split("/")[-1].split("_")[2] for run in run_list])}.eps', dpi=1200, format='eps', bbox_inches='tight')
    plt.clf()


def plot_temperature(data_folder, plot_dir, x_axis, fontsize=14):
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    run_list = os.listdir(data_folder)
    plt.figure(figsize=(5, 3))
    for run in run_list:
        run_files = os.listdir(f'{data_folder}/{run}')
        sr_list = sorted(run_files, key=lambda x: int(x.split('.')[0].split('sr')[-1]))
        for sr_file in sr_list:
            with open(f'{data_folder}/{run}/{sr_file}', 'r') as f:
                sr_data = f.readline().strip().split(' ')
                sr_data = list(map(lambda x: float(x), sr_data))
                max_point = np.argmax(sr_data)
                with open(f'run/temperature/best_temp/{run}.txt', 'a') as ff:
                    ff.write(f'{x_axis[max_point]} ')
                plt.plot(x_axis, sr_data, '--', linewidth=1, label=sr_file.split('.')[0])
                plt.plot(x_axis[max_point], sr_data[max_point], 'ro', markersize=4)
        # plt.title(f'Accuracy vs. Scaling Factor (iteration=20) \n ({run.split("_")[-3].split("-")[-1]})', fontsize=fontsize)
        # plt.xlabel('Scaling Factor', fontsize=fontsize)
        plt.ylabel('SAT Accuracy (%)', fontsize=fontsize)

        handles, labels = plt.gca().get_legend_handles_labels()
        sorted_dict = {}
        sorting = dict(zip(labels, handles))
        sorting = sorted(sorting.items(), key=lambda kv: kv[0])
        for item in sorting:
            sorted_dict[item[0]] = item[1]
        labels = sorted_dict.keys()
        handles = sorted_dict.values()
        plt.legend(handles, labels, labelspacing=0.01, framealpha=0, fontsize=fontsize)

        plt.savefig(f'{plot_dir}/temperature_{"_".join([run.split("/")[-1].split("_")[2]])}.eps', dpi=1200, format='eps', bbox_inches='tight')
        plt.clf()


def plot_loss(model_folder, plot_dir):
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    loss_file_list = []
    for item in os.listdir(model_folder):
        if 'npy' in item and '1015' in item:
            loss_file_list.append(item)
    for item in loss_file_list:
        file_dir = f'{model_folder}{item}'
        loss = np.load(file_dir)
        plt.plot(loss, linewidth=1)
        plt.title(f'Training Loss \n ({"_".join(item.split("_")[:3])})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'{plot_dir}/loss_{"_".join(item.split("_")[:3])}.png', bbox_inches='tight')
        plt.clf()


if __name__ == '__main__':
    # data_folder = f'run/solve/for_plot'
    # plot_dir = f'run/solve/plots'

    data_folder = f'run/solve_1009/for_plot/'
    # plot_dir = f'run/res/simulation/sat'
    plot_dir = f'run/res/simulation/assignment'

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # plot_merge('ep50_round24_neurosat', 'ep50_round24_nnsat', 'run/test/plot_arch', 'run/res', [8, 16,32,64,128])
    # plot_merge('ep50_round24_neurosat-lstm_1001_g1', 'ep50_round24_nnsat-lstm_1001_g1', 'run/test_1015/for_plot', 'run/res', [60,70,80,90])

    # plot_one(f'run/test/plot_arch/ep50_round24_nnsat', f'run/res/')
    # plot_merge('ep50_round24_neurosat', 'ep50_round24_nnsat', f'run/test/plot_arch/', f'run/res/', [8, 16, 32, 64, 128])

    # plot_all(data_folder, plot_dir)
    # plot_merge('ep50_round24_neurosat_0928_g1', 'ep50_round24_nnsat_0928_g1', data_folder, plot_dir, [10, 20, 30, 40, 50, 60])
    # plot_merge('ep50_round24_neurosat_0928_g1', 'ep50_round24_ggcn_0930', data_folder, plot_dir, [10, 20, 30, 40, 50, 60])
    # plot_merge('ep50_round24_ggcn_0930', 'ep50_round24_nnsat-ggcn_0930', data_folder, plot_dir, [10, 20, 30, 40, 50, 60])
    # plot_merge('ep50_round24_neurosat-lstm_1001_g1', 'ep50_round24_nnsat-lstm_1001_g1', f'run/test_circuit/for_plot/', f'run/test_circuit/plots', [10, 20, 30, 40, 50, 60])
    # plot_merge('ep50_round24_ggcn-lstm_1001_g1', 'ep50_round24_nnsat-ggcn-lstm_1001_g1', data_folder, plot_dir,
    #            [10, 20, 30, 40, 50, 60])

    # fl = os.listdir(data_folder)
    # target_fl = []
    # for item in fl:
    #     if '1001' in item:
    #         target_fl.append(item)
    # rnn_type = ['rnn', 'gru', 'lstm']
    # baselines = ['neurosat', 'ggcn']
    # model_list = []
    # for b in baselines:
    #     for r in rnn_type:
    #         model_list.append(f'{b}-{r}')
    # for m in model_list:
    #     plot_merge(f'ep50_round24_{m}_1001_g1', f'ep50_round24_nnsat-{m if "neurosat" not in m else m.split("-")[1]}_1001_g1', data_folder, plot_dir,
    #                [10, 20, 30, 40, 50, 60])

    temperature_dir = f'run/temperature/for_plot/'
    temperature_plot_dir = f'run/res/simulation/temperature'

    plot_temperature(temperature_dir, temperature_plot_dir, np.arange(1, 11.5, 0.5))

    # loss_dir = f'model/group_12_circuit/'
    # loss_dir_plot_dir = f'model/group_12_circuit/plots'
    # plot_loss(loss_dir, loss_dir_plot_dir)
