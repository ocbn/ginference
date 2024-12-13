from statistics import mean
import sklweka.jvm as jvm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklweka.classifiers import WekaEstimator
from sklweka.dataset import load_arff, to_nominal_labels
from sklweka.preprocessing import WekaTransformer
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


jvm.start(packages=True)

# ----------------------------example usage of sklearn-weka-plugin package without feature selection-----------------
# path = "../data/iris.arff"
# X, y, meta = load_arff(path, class_index="last")
# y = to_nominal_labels(y)
# j48 = WekaEstimator(classname="weka.classifiers.trees.J48")
# j48.fit(X, y)
# scores = j48.predict(X)
# probas = j48.predict_proba(X)
# print("\nJ48 on iris\nactual label -> predicted label, probabilities")
# for i in range(len(y)):
#     print(y[i], "->", scores[i], probas[i])
# -------------------example usage of sklearn-weka-plugin package with cv and without feature selection-----------------
# path = "../data/iris.arff"
# X, y, meta = load_arff(path, class_index="last")
# y = to_nominal_labels(y)
# # y = label_encoder.fit_transform(y)
# j48 = WekaEstimator(classname="weka.classifiers.trees.J48")
# j48 = RandomForestClassifier(random_state=1)
# cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
# accuracy_scores = cross_val_score(j48, X, y, cv=cv, scoring='accuracy')
# print("-------------------------------------------------------")
# print(accuracy_scores)
# print("-------------------------------------------------------")
# print('Mean Accuracy: %.3f' % mean(accuracy_scores))
# ----------------------- sklearn-weka-plugin with feature selection and cv-------------------------------------------
# path = "../data/iris.arff"
# X, y, meta = load_arff(path, class_index="last")
# y = to_nominal_labels(y)
# tr = WekaTransformer(classname="weka.filters.supervised.attribute.PartitionMembership")
# X_new, y_new = tr.fit(X, y).transform(X, y)
# j48 = WekaEstimator(classname="weka.classifiers.trees.J48")
# accuracy_scores = cross_val_score(j48, X_new, y_new, cv=10, scoring='accuracy')
# print("-------------------------------------------------------")
# print(accuracy_scores)
# print("-------------------------------------------------------")
# print('Mean Accuracy: %.3f' % mean(accuracy_scores))


# example usage of python-weka-wrapper package! produces the same results with cv on whole reduced data with GUI!
# --------------------------------------------classification with 10-fold cross-validation-------------------------
# from weka.core.converters import Loader
# import weka.core.converters as converters
# from weka.classifiers import Evaluation
# from weka.core.classes import Random
# from weka.filters import Filter
# from weka.classifiers import Classifier
# loader = Loader(classname="weka.core.converters.ArffLoader")
# data = loader.load_file(path)
# data.class_is_last()

# remove = Filter(classname="weka.filters.supervised.attribute.PartitionMembership")
# remove.inputformat(data)
# new_data = remove.filter(data)
#
# cls = Classifier(classname="weka.classifiers.trees.J48")
# evl = Evaluation(new_data)
# evl.crossvalidate_model(cls, new_data, 10, Random(1))
#
# print(evl.percent_correct)
# print(evl.summary())
# print(evl.class_details())
# print(evl.confusion_matrix)
# --------------------------classification on test data without feature selection-------------------------------------
# train, test = data.train_test_split(80, Random(1))
# converters.save_any_file(train, "train_iris.arff")
# converters.save_any_file(test, "test_iris.arff")
# cls.build_classifier(train)
# evl = Evaluation(test)
# evl.test_model(cls, test)
# print(evl.summary())
# print(evl.class_details())
# print(evl.confusion_matrix)
# --------------------------classification on test data with feature selection-------------------------------------
# train, test = data.train_test_split(70, Random(1))
# remove = Filter(classname="weka.filters.supervised.attribute.PartitionMembership")
# remove.inputformat(train)
# print(train.num_attributes)
# new_train = remove.filter(train)
# new_test = remove.filter(test)
# # converters.save_any_file(new_train, "train_iris.arff")
# # converters.save_any_file(new_test, "test_iris.arff")
# cls = Classifier(classname="weka.classifiers.trees.J48")
# cls.build_classifier(new_train)
# evl = Evaluation(new_test)
# evl.test_model(cls, new_test)
# print(new_train.num_attributes)
# print(evl.summary())
# print(evl.class_details())
# print(evl.confusion_matrix)
filter_options = ['-W', 'weka.classifiers.trees.RandomForest', '--', '-P', '100', '-I', '34']
tr = WekaTransformer(classname="weka.filters.supervised.attribute.PartitionMembership", options=filter_options)
print(tr.options)
jvm.stop()
