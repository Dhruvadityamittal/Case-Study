from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis


def get_model(classifier_index):
    classifiers = [
        LogisticRegression(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        KNeighborsClassifier(),   
        AdaBoostClassifier(),
        QuadraticDiscriminantAnalysis(),
        LinearDiscriminantAnalysis()
    ]
    try:
        classifier = classifiers[classifier_index]
        return classifier
    except:
        print("Print Select from these Classifier Indices where Classifer Index Map is :\n")
        for index, classifier in enumerate(classifiers) : print(f"{index} -> {classifier.__class__.__name__}")
        exit()
