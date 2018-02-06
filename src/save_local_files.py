import os

import pandas as pd

def save_submission(name, result, path='results'):

    path += '/submissions'

    if not os.path.exists(path):
        os.mkdir(path)

    name_base = '{:s}_submission_'.format(name)

    index = 0
    if os.listdir(os.path.join(path)):
        indexes = sorted([int(f.split(name_base)[1].split('.csv')[0]) for f in os.listdir(path) if name_base in f], reverse=True)
        if indexes:
            index = indexes[0] + 1
    
    file_name = name_base + str(index) + '.csv'

    with open(os.path.join(path, file_name), 'w') as file:
        file.write('Id,Scene_label\n')

        for n, label in enumerate(result):
            file.write('{:d},{:s}\n'.format(n, label))


def save_analysis(model_name, method_name, n, data, path='results', expand=False):

    path += '/analysis_data'

    if not os.path.exists(path):
        os.mkdir(path)

    method_name = method_name.replace('_', '-')

    name_base = '{:s}_n{:d}_{:s}_analysis_'.format(model_name, n, method_name)

    index = 0
    if os.listdir(os.path.join(path)):
        indexes = sorted([int(f.split(name_base)[1].split('.csv')[0]) for f in os.listdir(path) if name_base in f], reverse=True)
        if indexes:
            index = indexes[0] + 1

    file_name = name_base + str(index) + '.csv'

    if not expand and index > 0:
        return

    data.to_csv(os.path.join(path, file_name), sep=',')
