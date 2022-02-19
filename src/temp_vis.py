import os
import numpy as np


def temp_decrease():
    temperature_decrease_dir = f'model/group_12_circuit'
    temp_file_list = []
    for item in os.listdir(temperature_decrease_dir):
        if '.npy' in item:
            temp_file_list.append(item)
            print(item)
    temp_file_list = [temp_file_list[2], temp_file_list[0], temp_file_list[1]]
    print(temp_file_list)
    data = []
    for item in temp_file_list:
        data.append(np.load(os.path.join(temperature_decrease_dir, item)))
    data = np.array(data)
    data = data[:, [0, 9, 19, 29, 39, 49]].T
    for idx, item in enumerate(data):
        item = list(map(lambda x: format(x, '.3f'), item.tolist()))
        print(f'{idx * 10} & {" & ".join(item)} \\\\')

def opt_temp():
    test_temp = f'run/test_temperature/for_plot'
    neurosat_dir = f'run/test_1009/for_plot'
    best_temp_dir = f'run/temperature/best_temp'
                    
    temp_record = {}
    for item in os.listdir(test_temp):
        if 'ggcn' not in item:
            with open(os.path.join(test_temp, item, 'temperature_iter20.txt'), 'r') as f:
                line = f.readline().strip().split(' ')
            temp_record[item.split('_')[-3].split('-')[-1]] = {
                'scal': list(map(lambda x: format(float(x), '.3f'), line))
            }
            # print(list(map(lambda x: format(float(x), '.3f'), line)))

    for item in os.listdir(neurosat_dir):
        if 'neurosat' in item:
            base_acc = []
            sr_list = os.listdir(os.path.join(neurosat_dir, item))
            sr_list = sorted(sr_list, key=lambda x: int(x.split('.')[0].split('sr')[-1]))
            for sr in sr_list:
                with open(os.path.join(neurosat_dir, item, sr), 'r') as f:
                    base_acc.append(format(float(f.readline().strip().split(' ')[1]), '.3f'))
            # print(base_acc)
            temp_record[item.split('_')[-3].split('-')[-1]]['base'] = base_acc

    for item in os.listdir(best_temp_dir):
        with open(os.path.join(best_temp_dir, item), 'r') as f:
            temp_record[item.split('_')[-3].split('-')[-1]]['temp'] = f.readline().strip().split(' ')
    # print(temp_record)

    matrix = []
    for item in ['lstm', 'rnn', 'gru']:
        matrix.append(temp_record[item]['temp'])
        matrix.append(temp_record[item]['base'])
        matrix.append(temp_record[item]['scal'])
    matrix = np.array(matrix)
    matrix = matrix.T

    for idx, sr in enumerate(range(30, 81, 10)):
        
        template = '''
\multicolumn{1}{c|}{%s}&
\multicolumn{1}{c}{%s} & %s & %s &
\multicolumn{1}{c}{%s} & %s & %s &
\multicolumn{1}{c}{%s} & %s & %s \\\\
        ''' % (sr, matrix[idx][0], matrix[idx][1], matrix[idx][2], matrix[idx][3], matrix[idx][4], matrix[idx][5], matrix[idx][6], matrix[idx][7], matrix[idx][8])
        print(template)

def circuit_tab():
    circuit_dir = f'run/test_1015/for_plot'
    # circuit_dir = f'run/test_circuit/for_plot'
    data = {}
    for item in os.listdir(circuit_dir):
        if '1001' in item:
            sr_list = os.listdir(os.path.join(circuit_dir, item))
            sr_list = sorted(sr_list, key=lambda x: int(x.split('.')[0].split('sr')[-1]))
            for sr in sr_list:
                if sr not in data:
                    data[sr] = {}
                with open(os.path.join(circuit_dir, item, sr), 'r') as f:
                    data[sr][item.split('_')[-3].split('-')[0]] = list(map(lambda x: format(float(x), '.3f'), f.readline().strip().split(' ')))[:4]
    matrix = []
    for item in data:
        matrix.append(data[item]['neurosat'])
        matrix.append(data[item]['nnsat'])
    matrix = np.array(matrix).T
    for idx, iter in enumerate(matrix):
#         template = '''
# \multicolumn{1}{c|}{%s}&
# \multicolumn{1}{c}{%s} & %s & 
# \multicolumn{1}{c}{%s} & %s & 
# \multicolumn{1}{c}{%s} & %s & 
# \multicolumn{1}{c}{%s} & %s &
# \multicolumn{1}{c}{%s} & %s &
# \multicolumn{1}{c}{%s} & %s \\\\
#         ''' % ((idx + 1) * 10, iter[0], iter[1], iter[2], iter[3], iter[4], iter[5], iter[6], iter[7], iter[8], iter[9], iter[10], iter[11])
        template = '''
\multicolumn{1}{c|}{%s}&
\multicolumn{1}{c}{%s} & %s & 
\multicolumn{1}{c}{%s} & %s & 
\multicolumn{1}{c}{%s} & %s & 
\multicolumn{1}{c}{%s} & %s \\\\
        ''' % ((idx + 1) * 10, iter[0], iter[1], iter[2], iter[3], iter[4], iter[5], iter[6], iter[7])
        print(template)

if __name__ == '__main__':
    # opt_temp()
    circuit_tab()
