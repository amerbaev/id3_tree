import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from id3 import id3, predict, count_nodes

if __name__ == '__main__':
    ttt_df = pd.read_csv('car.data', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
    target = np.array(ttt_df['class'].values)
    ttt_df.drop('class', axis=1, inplace=True)
    feature_values = np.array([set(ttt_df[column].values) for column in ttt_df])
    feature_names = np.array(ttt_df.columns.values)
    data = ttt_df.values

    for limit in range(6, 3, -1):
        print(limit)
        cnt = []
        err = []
        for i in range(300, len(data), 10):
            errors = []
            for _ in range(10):
                sl_data, sl_tgt = shuffle(data, target, n_samples=i)

                x_train, x_test, y_train, y_test = train_test_split(sl_data, sl_tgt, test_size=0.2)
                tree = id3(x_train, y_train, feature_names, feature_values, limit)
                print(count_nodes(tree))

                # pretty_print_tree(tree)
                y_pred = predict(tree, feature_names, x_test)
                error = 0
                for p, t in zip(y_pred, y_test):
                    if p != t:
                        error += 1
                errors.append(error/len(y_pred))
            cnt.append(i)
            err.append(sum(errors)/len(errors))

        plt.figure()
        plt.plot(cnt, err)
        plt.show()
