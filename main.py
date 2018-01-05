import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from id3 import id3, predict, properties, pretty_print_tree, count_nodes

if __name__ == '__main__':
    ttt_df = pd.read_csv('car.data', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
    target = np.array(ttt_df['class'].values)
    ttt_df.drop('class', axis=1, inplace=True)
    feature_names = np.array(ttt_df.columns.values)
    data = ttt_df.values

    data, target = shuffle(data, target)
    # x_learn, x_valid, y_learn, y_valid = train_test_split(data, target, test_size=0.3)
    root = id3(data, target, feature_names)
    pretty_print_tree(root)
    print(count_nodes(root))
