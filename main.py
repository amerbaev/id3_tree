import pandas as pd
import numpy as np

from id3 import id3, pretty_print_tree

if __name__ == '__main__':
    ttt_df = pd.read_csv('tic-tac-toe.csv', names=[str(i) for i in range(1, 10)] + ['result'])
    target_names = np.array(['negative', 'positive'])
    target = np.array([1 if v == 'positive' else 0 for v in ttt_df['result'].values])
    ttt_df.drop('result', axis=1, inplace=True)
    feature_names = np.array(ttt_df.columns.values)
    data = ttt_df.values

    tree = id3(data, target, feature_names, target_names)
    pretty_print_tree(tree)