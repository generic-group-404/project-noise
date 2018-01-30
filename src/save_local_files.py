import os

def save_result(name, result, path='results'):

    if not os.path.exists(path):
        os.mkdir(path)

    name_base = '{:s}_submission_'.format(name)

    index = 0
    if os.listdir(os.path.join(path)):
        indexes = sorted([int(f.split(name_base)[1].split('.csv')[0]) for f in os.listdir(path) if name in f], reverse=True)
        if indexes:
            index = indexes[0] + 1
    
    file_name = name_base + str(index) + '.csv'

    with open(os.path.join(path, file_name), 'w') as file:
        file.write('Id,Scene_label\n')

        for n, label in enumerate(result):
            file.write('{:d},{:s}\n'.format(n, label))
