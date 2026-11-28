# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

from dmba import classificationSummary 

# begin by setting seed

np.random.seed(1)

# read the right worksheet from excel
data = pd.read_excel("ebayAuctions.xlsx", sheet_name="eBay Auctions")

# analyze size of observations with null values
print(data.shape)
data.info()

# nonnull values = rows for all columns so we proceed with data processing
# we begin by creating dummy variables and removing original categorical columns as requested

dummy_columns = ["Category", "Currency", "endDay", "Duration"]
for column in dummy_columns:
    if data[column].dtype != "string":
        data[column] = data[column].astype(str)
    
data = pd.get_dummies(data, columns = dummy_columns, drop_first=True)

# we now evaluate the numerical columns and check for their distributions and outliers
# Since we are working on a decision tree, the tree will find logical splits and we can keep them as numbers
    # Though I did realize this late and made a rudimentary splitting based on histograms
# Let's ensure that they are numbers

numerical_columns = ["sellerRating", "OpenPrice", "ClosePrice"]
for column in numerical_columns:
    if data[column].dtype != "float64" and data[column].dtype != "int64":
        data[column] = pd.to_numeric(data[column], errors='coerce')

# We can actually drop the closeprice since we will not have this in the incoming data
# turns out we do this later in the asignment - will keep here for now, and give data different name
data_noclose = data.drop(columns=["ClosePrice"])

# Let's now set the target variable to be boolean and move it to the end

data["Competitive?"] = data["Competitive?"].astype(bool)
data["Competitive?"] = data.pop("Competitive?")

# Double check
# print(data.shape)
# data.info()

# Data processing complete, ready for modeling
# Creating splits

X = data.drop(columns=['Competitive?'])
y = data['Competitive?']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# print("Training set size:", X_train.shape)
# print("Test set size:", X_test.shape)

# Creating decision tree, min_samples_split = 50
fullClassTree = DecisionTreeClassifier(min_samples_split=50, random_state = 1)
fullClassTree.fit(X_train, y_train)

#evaluation train accuracy
print("Training Accuracy (WITH CLOSE PRICE):")
classificationSummary(y_train, fullClassTree.predict(X_train))
# (8:43 pm 24 Nov) 88% - open to suggestions on how to improve accuracy

# visualizing the tree
print("Decision Tree Representation (WITH CLOSE PRICE):")
feature_names = list(X.columns)
text_representation = tree.export_text(fullClassTree, feature_names = feature_names)
print(text_representation)

# Let's now use data_noclose to compare accuracy

data_noclose["Competitive?"] = data_noclose["Competitive?"].astype(bool)
data_noclose["Competitive?"] = data_noclose.pop("Competitive?")

# Double check
# print(data_noclose.shape)
# data_noclose.info()

# Data processing complete, ready for modeling
# Creating splits

X_noclose = data_noclose.drop(columns=['Competitive?'])
y_noclose = data_noclose['Competitive?']
X_train, X_test, y_train, y_test = train_test_split(X_noclose, y_noclose, test_size=0.4, random_state=1)

# print("Training set size:", X_train.shape)
# print("Test set size:", X_test.shape)

# Creating decision tree, min_samples_split = 50
fullClassTree = DecisionTreeClassifier(min_samples_split=50, random_state = 1)
fullClassTree.fit(X_train, y_train)

#evaluation train accuracy
print("Training Accuracy (WITHOUT CLOSE PRICE):")
classificationSummary(y_train, fullClassTree.predict(X_train))
# (8:43 pm 24 Nov) 88% - open to suggestions on how to improve accuracy

# visualizing the tree
print("Decision Tree Representation (WITHOUT CLOSE PRICE):")
feature_names = list(X_noclose.columns)
text_representation = tree.export_text(fullClassTree, feature_names = feature_names)
print(text_representation)