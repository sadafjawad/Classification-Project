#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB

# reading the data
data = pd.read_csv(r'/Users/adamozdemir/Desktop/CPS844/A1/6_class_csv.csv')

# preprocessing and splitting
X = data.drop('Star type', axis=1)
label_encoder = LabelEncoder()
# change encoding of star color because data is in string format
X['star_color_encoded'] = label_encoder.fit_transform(X['Star color'])
X = X.drop('Star color', axis=1)
# change encoding of spectral class because data is in string format
X['spectral_class_encoded'] = label_encoder.fit_transform(X['Spectral Class'])
X = X.drop('Spectral Class', axis=1)
Y = data['Star type']
xTrain, xTest, yTrain, yTest = train_test_split(
    X, Y, test_size=0.3, random_state=42)

# Decision tree classifier
myTree = tree.DecisionTreeClassifier(
    criterion='entropy', max_depth=3, random_state=42)
myTree.fit(xTrain, yTrain)
test_predY = myTree.predict(xTest)
test_accuracy = accuracy_score(yTest, test_predY)
print(f"Accuracy of the Tree classifier on testData: {test_accuracy}")
plt.figure(figsize=(20, 10), dpi=600) 
plot_tree(myTree, feature_names=list(xTrain.columns),
          class_names=list(map(str, myTree.classes_)), filled=True)
plt.show()
importancesTree = myTree.feature_importances_
for feature_name, importance in zip(X.columns, importancesTree):
    print(f"{feature_name}: {importance:.4f}")
print("\n")

# K Nearest Neighbor Classifier
knn = KNeighborsClassifier(n_neighbors=3)
# trains the classifier
knn.fit(xTrain, yTrain)
# predicts the test data
test_predY_knn = knn.predict(xTest)
# calculate accuracy and assign it to a new variable
test_accuracy_knn = accuracy_score(yTest, test_predY_knn)
print(f"Accuracy of the KNN classifier on test data: {test_accuracy_knn}")
print(f"In KNN classifier all attributes are equally important.")
print("\n")


# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=2, random_state=42)
# trains the random forest classifier on the training data
rf.fit(xTrain, yTrain)
# predict the labels for the test data using the trained classifier
test_predY_rf = rf.predict(xTest)
# calculate the accuracy of the test data
test_accuracy_rf = accuracy_score(yTest, test_predY_rf)
print(f"Accuracy of the Random Forest classifier on testData: {test_accuracy_rf}")
importancesForest = rf.feature_importances_
importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': importancesForest})
print(importances_df.to_string(index=False))


# Bagging Classifier
base_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
# creates a bagging classifier
bagging_classifier = BaggingClassifier(estimator=base_tree, n_estimators=5, random_state=42)
# train the bagging classifier
bagging_classifier.fit(xTrain, yTrain)
# predict the test set
test_predY_bagging = bagging_classifier.predict(xTest)
# calculate accuracy
test_accuracy_bagging = accuracy_score(yTest, test_predY_bagging)
print(f"Accuracy of the bagging classifier on test data: {test_accuracy_bagging}")
# compute the mean feature importances across all trees in the bagging classifier
# store it in a variable
feature_importances = np.mean([
    tree.feature_importances_ for tree in bagging_classifier.estimators_
    ], axis=0)
# iterate through each feature and its corresponding importance and print them
for feature_name, importance in zip(xTrain.columns, feature_importances):
    print(f"{feature_name}: {importance:.4f}")
print("\n")

# Naive Bayes classifier
nb_classifier = GaussianNB()
# trains the classifier using xTrain and yTrain
nb_classifier.fit(xTrain, yTrain)
# predicts the test set
test_predY_nb = nb_classifier.predict(xTest)
# calculate the accuracy of the test set
test_accuracy_nb = accuracy_score(yTest, test_predY_nb)
print(f"Accuracy of the Naive Bayes classifier on test data: {test_accuracy_nb}")
# calculates the variance for each feature
feature_variances = np.var(xTrain, axis=0)
# sort the variances and get the indices in ORDER of lowest to largest variance
sort_indices = np.argsort(feature_variances)
# display the sorted variances with their corresponding feature names
print("Features sorted by importance (based on variance):")
# iterate through the indices of sorted variances
for index in sort_indices:
    # get the feature name using the index
    feature_name = xTrain.columns[index]
    # get the variance of the feature
    variance = feature_variances[feature_name]
    # display the feature name and its corresponding variance
    print(f"{feature_name}: Variance = {variance:.4f}")
