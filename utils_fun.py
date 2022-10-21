import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree


class GetTitanicDataset:
    def __init__(self, df: pd.DataFrame = None, **kwargs):
        self.df = df
        self.path = kwargs.get('path')
        self.col_to_drop = kwargs.get('col_to_drop')
        self.col_dtype = kwargs.get('col_dtype')

        if not self.df:
            self.get_data()

    def __call__(self):
        self.prepare_dataset()
        self.change_dtype()
        return self.get_data_target()

    def get_data(self):
        if not self.path:
            self.path = "https://raw.githubusercontent.com/Kamil128/Learning_repo/main/ML_Classification/data/titanic_train.csv"
        self.df = pd.read_csv(self.path)

    def prepare_dataset(self):
        if not self.col_to_drop:
            self.col_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        self.df.drop(self.col_to_drop, axis=1, inplace=True)

    def change_dtype(self):

        if not self.col_dtype:
            self.col_dtype = {
                'Sex': 'category',
                'Age': 'int',
                'Embarked': 'category',
                'Survived': 'category',
                'Pclass': 'category',
            }

        for col, dtype in self.col_dtype.items():
            self.df[col] = self.df[col].astype(dtype, errors='ignore')

    def get_data_target(self):
        return self.df.drop(['Survived'], axis=1), self.df['Survived']


def show_decision_path(estimator):
    clf = estimator

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    for i in range(n_nodes):
        if is_leaves[i]:
            print(
                "{space}node={node} is a leaf node.".format(
                    space=node_depth[i] * "\t", node=i
                )
            )
        else:
            print(
                "{space}node={node} is a split node: "
                "go to node {left} if X[:, {feature}] <= {threshold} "
                "else to node {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i],
                )
            )

    ##############################################################################
    # We can compare the above output to the plot of the decision tree.

    plt.figure(figsize=(15, 7))
    tree.plot_tree(clf, node_ids=True)
    plt.show()


def show_decision_sample(estimator, x, transformer=None, sample_id=0):

    if transformer:
        x = transformer.transform(x)

    clf = estimator

    node_indicator = clf.decision_path(x)
    leaf_id = clf.apply(x)
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    n_nodes = clf.tree_.node_count

    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[
                 node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
                 ]

    print("Rules used to predict sample {id}:\n".format(id=sample_id))
    for node_id in node_index:
        # continue to the next node if it is a leaf node
        if leaf_id[sample_id] == node_id:
            continue

        # check if value of the split feature for sample 0 is below threshold
        if x[sample_id, feature[node_id]] <= threshold[node_id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print(
            "decision node {node} : (x[{sample}, {feature}] = {value}) "
            "{inequality} {threshold})".format(
                node=node_id,
                sample=sample_id,
                feature=feature[node_id],
                value=x[sample_id, feature[node_id]],
                inequality=threshold_sign,
                threshold=threshold[node_id],
            )
        )

    ##############################################################################
    # For a group of samples, we can determine the common nodes the samples go
    # through.

    sample_ids = [0, 1]
    # boolean array indicating the nodes both samples go through
    common_nodes = node_indicator.toarray()[sample_ids].sum(axis=0) == len(sample_ids)
    # obtain node ids using position in array
    common_node_id = np.arange(n_nodes)[common_nodes]

    print(
        "\nThe following samples {samples} share the node(s) {nodes} in the tree.".format(
            samples=sample_ids, nodes=common_node_id
        )
    )
    print("This is {prop}% of all nodes.".format(prop=100 * len(common_node_id) / n_nodes))
