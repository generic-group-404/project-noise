import os

import pandas as pd

def save_submission(name, result, path='results'):

    path += '/submissions'

    if not os.path.exists(path):
        os.mkdir(path)

    name_base = '{:s}_submission_'.format(name)

    index = find_file_index(path, name_base)
    
    file_name = name_base + str(index) + '.csv'

    with open(os.path.join(path, file_name), 'w') as file:
        file.write('Id,Scene_label\n')

        for n, label in enumerate(result):
            file.write('{:d},{:s}\n'.format(n, label))


def save_analysis(name, data, *tags, path='results', folder='/analysis_data', expand=True, form='.csv'):

    if not os.path.exists(path):
        os.mkdir(path)

    path += folder

    if not os.path.exists(path):
        os.mkdir(path)

    name_base = '{:s}_analysis_{:s}_'.format(name, '_'.join(tags))

    index = find_file_index(path, name_base)

    file_name = name_base + str(index) + '.csv'

    if not expand and index > 0:
        return

    data.to_csv(os.path.join(path, file_name), sep=',')


def get_fig_file(name, *tags, path='figures'):

    if not os.path.exists(path):
        os.mkdir(path)
    
    name_base = '{:s}_{:s}_'.format(name, '_'.join(tags))

    index = find_file_index(path, name_base, form='.png')

    return os.path.join(path, (name_base + str(index) + '.png'))


def find_file_index(path, name, form='.csv'):
    """Finds the current index for file"""
    index = 0
    if os.listdir(os.path.join(path)):
        indexes = sorted([int(f.split(name)[1].split(form)[0]) for f in os.listdir(path) if name in f], reverse=True)
        if indexes:
            index = indexes[0] + 1
    return index
