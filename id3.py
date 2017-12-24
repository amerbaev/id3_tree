import ast
import csv
import sys
import math
import os
from collections import Counter
import numpy as np


def get_header_name_to_idx_maps(headers):
    name_to_idx = {}
    idx_to_name = {}
    for i in range(0, len(headers)):
        name_to_idx[headers[i]] = i
        idx_to_name[i] = headers[i]
    return idx_to_name, name_to_idx


def project_columns(data, columns_to_project):
    data_h = list(data['header'])
    data_r = list(data['rows'])

    all_cols = list(range(0, len(data_h)))

    columns_to_project_ix = [data['name_to_idx'][name] for name in columns_to_project]
    columns_to_remove = [cidx for cidx in all_cols if cidx not in columns_to_project_ix]

    for delc in sorted(columns_to_remove, reverse=True):
        del data_h[delc]
        for r in data_r:
            del r[delc]

    idx_to_name, name_to_idx = get_header_name_to_idx_maps(data_h)

    return {'header': data_h, 'rows': data_r,
            'name_to_idx': name_to_idx,
            'idx_to_name': idx_to_name}


def get_uniq_values(data):
    idx_to_name = data['idx_to_name']
    idxs = idx_to_name.keys()

    val_map = {}
    for idx in iter(idxs):
        val_map[idx_to_name[idx]] = set()

    for data_row in data['rows']:
        for idx in idx_to_name.keys():
            att_name = idx_to_name[idx]
            val = data_row[idx]
            if val not in val_map.keys():
                val_map[att_name].add(val)
    return val_map


def get_class_labels(data, target_attribute):
    rows = data['rows']
    col_idx = data['name_to_idx'][target_attribute]
    labels = {}
    for r in rows:
        val = r[col_idx]
        if val in labels:
            labels[val] = labels[val] + 1
        else:
            labels[val] = 1
    return labels


def entropy(target):
    ent = 0
    for label in set(target):
        p_x = len([t for t in target if t == label]) / len(target)
        ent += - p_x * math.log(p_x, 2)
    return ent


def partition_data(data, group_att):
    partitions = {}
    data_rows = data['rows']
    partition_att_idx = data['name_to_idx'][group_att]
    for row in data_rows:
        row_val = row[partition_att_idx]
        if row_val not in partitions.keys():
            partitions[row_val] = {
                'name_to_idx': data['name_to_idx'],
                'idx_to_name': data['idx_to_name'],
                'rows': list()
            }
        partitions[row_val]['rows'].append(row)
    return partitions


def sum_entropy_by_attr(data, index, values, target):
    # find uniq values of splitting att
    sum_ent = 0
    for val in values:
        tgt = target[data.T[index] == val]
        sum_ent += entropy(tgt)

    return sum_ent


def most_common_label(labels):
    mcl = max(labels, key=lambda k: labels[k])
    return mcl


def id3(data, target, features, target_names):
    node = {}

    if len(set(target)) == 1:
        node['label'] = target_names[target[0]]
        return node

    if len(features) == 0:
        node['label'] = target_names[Counter(target).most_common(1)[0]]
        return node

    ent = entropy(target)

    max_info_gain = None
    max_info_gain_index = None
    max_info_gain_values = None

    for index, column in enumerate(data.T):
        values = set(column)
        sum_ent = sum_entropy_by_attr(data, index, values, target)
        info_gain = ent - sum_ent
        if max_info_gain is None or info_gain > max_info_gain:
            max_info_gain = info_gain
            max_info_gain_index = index
            max_info_gain_values = values

    if max_info_gain is None:
        node['label'] = target_names[Counter(target).most_common(1)[0]]
        return node

    node['attribute'] = features[max_info_gain_index]
    node['nodes'] = {}

    data_for_subtrees = np.delete(data, max_info_gain_index, axis=1)
    features_for_subtrees = np.delete(features, max_info_gain_index)

    for att_value in max_info_gain_values:
        subtree_data = data_for_subtrees[data.T[max_info_gain_index] == att_value]
        subtree_target = target[data.T[max_info_gain_index] == att_value]
        node['nodes'][att_value] = id3(subtree_data, subtree_target, features_for_subtrees, target_names)

    return node


def load_config(config_file):
    with open(config_file, 'r') as myfile:
        data = myfile.read().replace('\n', '')
    return ast.literal_eval(data)


def pretty_print_tree(root):
    stack = []
    rules = set()

    def traverse(node, stack, rules):
        if 'label' in node:
            stack.append(' THEN ' + node['label'])
            rules.add(''.join(stack))
            stack.pop()
        elif 'attribute' in node:
            ifnd = 'IF ' if not stack else ' AND '
            stack.append(ifnd + node['attribute'] + ' EQUALS ')
            for subnode_key in node['nodes']:
                stack.append(subnode_key)
                traverse(node['nodes'][subnode_key], stack, rules)
                stack.pop()
            stack.pop()

    traverse(root, stack, rules)
    print(os.linesep.join(rules))


def main():

    target_attribute = config['target_attribute']
    remaining_attributes = set(data['header'])
    remaining_attributes.remove(target_attribute)

    uniqs = get_uniq_values(data)

    root = id3(data, uniqs, remaining_attributes, target_attribute)

    pretty_print_tree(root)


if __name__ == "__main__":
    main()
