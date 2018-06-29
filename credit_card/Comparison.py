import warnings 
warnings.filterwarnings('ignore')

import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier

class Comparison(object):
    """docstring for Comparison"""
    def __init__(self, data, target, features, scoring, record_file):
        super(Comparison, self).__init__()
        self.data = data
        self.target = target
        self.features = features
        self.scoring = scoring
        self.record_file = record_file


    def AmongModels(self):
        self.results = []
        self.names = []
        self.cost = []
        self.means = []
        self.stds = []
        
        models = []
        models.append(('LR', LogisticRegression()))
        # models.append(('SVC', LinearSVC(loss='hinge')))
        models.append(('SDG', SGDClassifier()))

        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('NB', GaussianNB()))

        models.append(('CART', DecisionTreeClassifier()))
        models.append(('RF', RandomForestClassifier()))
        models.append(('GBDT', GradientBoostingClassifier()))

        models.append(('NN', MLPClassifier()))
        with open(self.record_file, 'a') as file:
            file.write('\n'+'='*20+'\n')
        for name, model in models:
            start = time.time()
            kfold = model_selection.KFold(n_splits=10, random_state=0)
            cv_results = model_selection.cross_val_score(model, self.data[self.features], self.data[self.target], cv=kfold, scoring=self.scoring)
            time_cost = time.time()-start
            score_mean = cv_results.mean()
            score_std = cv_results.std()
            msg = "%s:\t%f (%f)\ttime: %f s" % (name, score_mean, score_std, time_cost)
            with open(self.record_file, 'a') as file:
                file.write(msg)
            print(msg)
            self.results.append(cv_results)
            self.names.append(name)
            self.means.append(score_mean)
            self.stds.append(score_std)
            self.cost.append(time_cost)

        self.AmongResults = pd.DataFrame(columns=['model', 'score_mean', 'score_std', 'time'])
        self.AmongResults['model'] = self.names
        self.AmongResults['score_mean'] = self.means
        self.AmongResults['score_std'] = self.stds
        self.AmongResults['time'] = self.cost

        return self.AmongResults

    def Visual(self, time=False):
        fig = plt.figure(figsize=(8, 8))
        if not time:        
            ax = fig.add_subplot(111)
            plt.boxplot(self.results)
            ax.set_xticklabels(self.names)
            plt.title('Algorithm Comparison')
        else:
            fig.suptitle('Algorithm Comparison')

            ax1=fig.add_subplot(111, label="1")
            ax2=fig.add_subplot(111, label="2", frame_on=False)

            ax1.errorbar(self.names, self.means, self.stds, color="C0", linestyle='None', marker='o')
            ax1.set_xlabel("model", color="C0")
            ax1.set_ylabel("score mean", color="C0")
            ax1.tick_params(axis="model", colors="C0")
            ax1.tick_params(axis="score mean", colors="C0")

            ax2.bar(self.names, self.cost, color="C1", alpha=0.3, width=0.5)
            ax2.xaxis.tick_top()
            ax2.yaxis.tick_right()
            ax2.set_xlabel('model', color="C1") 
            ax2.set_ylabel('time', color="C1")   
            ax2.xaxis.set_label_position('top') 
            ax2.yaxis.set_label_position('right') 
            ax2.tick_params(axis='model', colors="C1")
            ax2.tick_params(axis='time', colors="C1")
        plt.grid()
        plt.show()
        return None
        



