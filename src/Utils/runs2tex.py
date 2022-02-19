import os


def starT(iters, dis, runs, output):
    # for run in runs:
    iters = list(map(lambda x: str(x), iters))
    iterStr = f'({"/".join(iters)} iters)'
    temp = list(map(lambda x: x.split('.')[0], os.listdir(runs[0])))
    temp.sort(key=lambda x: int(x.split('sr')[1]))
    temp = list(map(lambda x: f'{x} {iterStr}', temp))
    title = ' & '.join(temp)

    row_list = []
    for run in runs:
        srs = os.listdir(run)
        srs.sort(key=lambda x: int(x.split('.')[0].split('sr')[1]))
        res_list = []
        for sr in srs:
            with open(os.path.join(run, sr), 'r') as f:
                line = f.readline().strip().split(' ')
                line = list(map(lambda x: format(float(x), f'.{dis}f'), line))
                line = '/'.join(line)
                res_list.append(line)
        if 'neurosat' in run:
            row_list.append(f'NeuroSAT & {" & ".join(res_list)} \\\\')
        if 'nnsat' in run and 'ggcn' not in run:
            row_list.append(f'NNSAT & {" & ".join(res_list)} \\\\')
    row_str = '\n'.join(row_list)
    print(row_str)

    templet = r'''
\begin{table*}[htbp]
\footnotesize
\centering
\caption{Add caption}
\resizebox{\textwidth}{6mm}{
\begin{tabular}{ c || c | c | c | c | c }
    \hline\hline
    Model & %title \\
    \hline\hline
    %rows
    \hline\hline
    \end{tabular}
}
\label{tab:neurosat_nnsat_c}
\end{table*}
    '''
    templet = templet.replace(r'%title', title)
    templet = templet.replace(r'%rows', row_str)
    print(templet)

    with open(output, 'w') as f:
        f.write(templet)


def improvement(basem_dir, newm_dir):
    sr_list = os.listdir(basem_dir)
    single_improvement = 0
    basem = []
    for sr in sr_list:
        with open(os.path.join(basem_dir, sr), 'r') as f:
            basem.append(list(map(lambda x: round(float(x), 3), f.readline().strip().split(' ')))[1:])
    newm = []
    for sr in sr_list:
        with open(os.path.join(newm_dir, sr), 'r') as f:
            newm.append(list(map(lambda x: round(float(x), 3), f.readline().strip().split(' ')))[1:])
    for i, line in enumerate(newm):
        for j, item in enumerate(line):
            single_improvement += (newm[i][j] - basem[i][j])
            # print((newm[i][j] - basem[i][j]))
    single_improvement = round((single_improvement / (len(newm) * len(newm[0]))), 3)
    print(single_improvement)


if __name__ == '__main__':
    run_dir = f'run/test/plot_arch'
    # runs = [
    #     os.path.join(run_dir, 'ep50_round24_neurosat'),
    #     os.path.join(run_dir, 'ep50_round24_nnsat')
    # ]

    # starT([8, 16, 32, 64, 128], 3, runs, f'run/res/tables/neurosat_nnsat_c.txt')
    improvement('./run/solve_1009/for_plot/ep50_round24_neurosat-lstm_1001_g1', './run/solve_1009/for_plot/ep50_round24_nnsat-lstm_1001_g1')
    improvement('./run/test_1009/for_plot/ep50_round24_neurosat-lstm_1001_g1', './run/test_1009/for_plot/ep50_round24_nnsat-lstm_1001_g1')
    improvement('./run/test_1009/for_plot/ep50_round24_ggcn-lstm_1001_g1', './run/test_1009/for_plot/ep50_round24_nnsat-ggcn-lstm_1001_g1')
