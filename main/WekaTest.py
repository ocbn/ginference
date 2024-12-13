import sklweka.jvm as jvm
from sklweka.classifiers import WekaEstimator
from sklweka.dataset import load_arff, to_nominal_labels

# start JVM with Weka package support
from sklweka.preprocessing import WekaTransformer

jvm.start(packages=True)

path = "../data/iris.arff"

# regression
# X, y, meta = load_arff(path, class_index="last")
# lr = WekaEstimator(classname="weka.classifiers.functions.LinearRegression")
# scores = cross_val_score(lr, X, y, cv=10, scoring='neg_root_mean_squared_error')
# print("Cross-validating LR on bolts (negRMSE)\n", scores)
#
# classification
# X, y, meta = load_arff(path, class_index="last")
# y = to_nominal_labels(y)
# j48 = WekaEstimator(classname="weka.classifiers.trees.J48", options=["-M", "3"])
# j48.fit(X, y)
# scores = j48.predict(X)
# probas = j48.predict_proba(X)
# print("\nJ48 on iris\nactual label -> predicted label, probabilities")
# for i in range(len(y)):
#     print(y[i], "->", scores[i], probas[i])
#
# # clustering
# X, y, meta = load_arff("/some/where/iris.arff", class_index="last")
# cl = WekaCluster(classname="weka.clusterers.SimpleKMeans", options=["-N", "3"])
# clusters = cl.fit_predict(X)
# print("\nSimpleKMeans on iris\nclass label -> cluster")
# for i in range(len(y)):
#     print(y[i], "->", clusters[i])
#
# # preprocessing
X, y, meta = load_arff(path, class_index="last")
tr = WekaTransformer(classname="weka.filters.unsupervised.attribute.Standardize", options=["-unset-class-temporarily"])
X_new, y_new = tr.fit(X, y).transform(X, y)
print("\nStandardize filter")
print("\ntransformed X:\n", X_new)
print("\ntransformed y:\n", y_new)
#
# # generate data
# gen = DataGenerator(
#     classname="weka.datagenerators.classifiers.classification.BayesNet",
#     options=["-S", "2", "-n", "10", "-C", "10"])
# X, y, X_names, y_name = generate_data(gen, att_names=True)
# print("X:", X_names)
# print(X)
# print("y:", y_name)
# print(y)
#
# # stop JVM
jvm.stop()
