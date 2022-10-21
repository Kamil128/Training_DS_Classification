from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    recall_score,
    precision_score,
    precision_recall_curve,
    RocCurveDisplay
)

import numpy as np

import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions


class MetricsUtils:
    def __init__(self, x_train, x_test, y_train, y_test, estimator):
        self.y_test = y_test
        self.x_train = x_train
        self.x_test = x_test
        self.estimator = estimator
        self.y_train = y_train

        self.y_pred = estimator.predict(self.x_test)
        self.y_train_pred = estimator.predict(self.x_train)

    def show_scores(self, test=True):
        if test:
            actual, pred = self.y_test, self.y_pred
        else:
            actual, pred = self.y_train, self.y_train_pred
        
        print(f"Accuracy: {accuracy_score(actual, pred):.2f}")
        print(f"Recall: {recall_score(actual, pred):.2f}")
        print(f"Precision: {precision_score(actual, pred):.2f}")
        print(f"Classification Report:\n{classification_report(actual, pred)}")

    def show_precision_recall(self):
        y_scores = self.estimator.predict_proba(self.x_train)[:, 1]
        precision, recall, threshold = precision_recall_curve(self.y_train, y_scores)

        plt.figure(figsize=(14, 7))

        plt.plot(threshold, precision[:-1], label='precision')
        plt.plot(threshold, recall[:-1], label='recall')
        # plt.axvline(x=0.5, color='black', linestyle='--')
        plt.legend()

        plt.show()

    def plot_roc_auc_curve(self):
        RocCurveDisplay.from_estimator(estimator=self.estimator,
                                       X=self.x_test,
                                       y=self.y_test)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.show()

    def plot_margin(self, model_name, scaler_name):
        x = self.x_train
        y = self.y_train
        x_test = self.x_test
        estimator = self.estimator[model_name]
        scaler = self.estimator[scaler_name]

        fig, ax = plt.subplots(figsize=(10, 7))
        plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        lin_x = np.linspace(xlim[0], xlim[1], 30)
        lin_y = np.linspace(ylim[0], ylim[1], 30)

        grid_Y, grid_X = np.meshgrid(lin_y, lin_x)

        xy = np.vstack([grid_X.ravel(), grid_Y.ravel()]).T

        xy = scaler.transform(xy)

        Z = estimator.decision_function(xy).reshape(grid_X.shape)

        ax.contour(grid_X, grid_Y, Z,
                   colors='k',
                   levels=[-1, 0, 1],
                   alpha=0.5,
                   linestyles=['--', '-', '--']
                   )

        support_vectors = estimator.support_vectors_

        support_vectors = scaler.inverse_transform(support_vectors)

        ax.scatter(support_vectors[:, 0], support_vectors[:, 1],
                   s=100, linewidth=3, facecolors='none', edgecolors='k')

        ax.scatter(x_test[:, 0], x_test[:, 1], marker='x', c=self.y_test)

        plt.show()

    def plot_boundary(self):
        plt.figure(figsize=(10, 7))
        plot_decision_regions(self.x_train,
                              self.y_train,
                              clf=self.estimator,
                              legend=2)
        plt.scatter(self.x_test[:, 0],
                    self.x_test[:, 1],
                    marker='x',
                    c=self.y_test)
        plt.show()
