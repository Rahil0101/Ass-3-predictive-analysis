import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import random

# Load the dataset
url = 'https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv'
data = pd.read_csv(url)

# Separate features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

ros = RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_resample(X, y)

# Create five samples using different sampling techniques
# (Systematic Sampling, Stratified Sampling, Cluster Sampling, Simple Random Sampling, and Random Undersampling)

# Systematic Sampling
# Set the seed for reproducibility
random.seed(0)
# Determine the step size using the formula (N/n)
step_size = int(len(X_train) / 100)  # assume 1% of the original dataset size
# Select the indices of the samples using the step size
indices_sys = list(range(0, len(X_train), step_size))
X_train_sys, y_train_sys = X_train.iloc[indices_sys], y_train.iloc[indices_sys]
# Split the data into training and test sets
X_train_sys, X_test_sys, y_train_sys, y_test_sys = train_test_split(X, y, test_size=0.3, random_state=0)

#Stratified Sampling
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, test_index in split.split(X_train, y_train):
    X_train_strat = X_train.iloc[train_index]
    X_test_strat = X_train.iloc[test_index]
    y_train_strat = y_train.iloc[train_index]
    y_test_strat = y_train.iloc[test_index]

# # Cluster Sampling
# # Set the seed for reproducibility
random.seed(0)
# # Determine the number of clusters using the formula sqrt(N)
n_clusters = int(len(X_train) ** 0.5)  # assume the square root of the original dataset size
# # Apply K-Means clustering to the training data
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(X_train)
# # Assign each sample to a cluster
X_train_cluster = X_train[kmeans.labels_ == 0]  # assume the first cluster
y_train_cluster = y_train[kmeans.labels_ == 0]
# # Split the data into training and test sets
X_train_cluster,X_test_cluster,y_train_cluster,y_test_cluster = train_test_split(X, y, test_size=0.3, random_state=0)

# # Simple Random Sampling
# # Set the seed for reproducibility
random.seed(0)
# Select a random sample of the training data
sample_size = 1000  # assume a sample size of 1000
indices_random = random.sample(range(len(X_train)), sample_size)
X_train_random, y_train_random = X_train.iloc[indices_random], y_train.iloc[indices_random]
# # Split the data into training and test sets
X_train_random,X_test_random,y_train_random, y_test_random = train_test_split(X, y, test_size=0.3, random_state=0)

# Define five different ML models
M1 = DecisionTreeClassifier(random_state=0)
M2 = RandomForestClassifier(random_state=0)
M3 = KNeighborsClassifier()
M4 = GaussianNB()
M5 = SVC(random_state=0)

# Fit each model on each of the five samples created using different sampling techniques
# and evaluate their performance on the test set


# M1
M1.fit(X_train_sys, y_train_sys)
y_pred_sys = M1.predict(X_test_sys)
print("M1 Accuracy (Systematic Sampling):", M1.score(X_test_sys,y_test_sys))
M1.fit(X_train_strat, y_train_strat)
y_pred_strat = M1.predict(X_test_strat)
print("M1 Accuracy (Stratified Sampling):", M1.score(X_test_strat,y_test_strat))
M1.fit(X_train_cluster, y_train_cluster)
y_pred_cluster = M1.predict(X_test_cluster)
print("M1 Accuracy (Cluster Sampling):", M1.score(X_test_cluster,y_test_cluster))
M1.fit(X_train_random, y_train_random)
y_pred_random = M1.predict(X_test_random)
print("M1 Accuracy (Simple Random Sampling):", M1.score(X_test_random,y_test_random))

# M2
M2.fit(X_train_sys, y_train_sys)
y_pred_sys = M2.predict(X_test_sys)
print("M2 Accuracy (Systematic Sampling):", M2.score(X_test_sys,y_test_sys))
M2.fit(X_train_strat, y_train_strat)
y_pred_strat = M2.predict(X_test_strat)
print("M2 Accuracy (Stratified Sampling):", M2.score(X_test_strat,y_test_strat))
M2.fit(X_train_cluster, y_train_cluster)
y_pred_cluster = M2.predict(X_test_cluster)
print("M2 Accuracy (Cluster Sampling):", M2.score(X_test_cluster,y_test_cluster))
M2.fit(X_train_random, y_train_random)
y_pred_random = M2.predict(X_test_random)
print("M2 Accuracy (Simple Random Sampling):", M2.score(X_test_random,y_test_random))

# M3
M3.fit(X_train_sys, y_train_sys)
y_pred_sys = M3.predict(X_test_sys)
print("M3 Accuracy (Systematic Sampling):", M3.score(X_test_sys,y_test_sys))
M3.fit(X_train_strat, y_train_strat)
y_pred_strat = M3.predict(X_test_strat)
print("M3 Accuracy (Stratified Sampling):", M3.score(X_test_strat,y_test_strat))
M3.fit(X_train_cluster, y_train_cluster)
y_pred_cluster = M3.predict(X_test_cluster)
print("M3 Accuracy (Cluster Sampling):", M3.score(X_test_cluster,y_test_cluster))
M3.fit(X_train_random, y_train_random)
y_pred_random = M3.predict(X_test_random)
print("M3 Accuracy (Simple Random Sampling):", M3.score(X_test_random,y_test_random))



# M4
M4.fit(X_train_sys, y_train_sys)
y_pred_sys = M4.predict(X_test_sys)
print("M4 Accuracy (Systematic Sampling):", M4.score(X_test_sys,y_test_sys))
M4.fit(X_train_strat, y_train_strat)
y_pred_strat = M4.predict(X_test_strat)
print("M4 Accuracy (Stratified Sampling):", M4.score(X_test_strat,y_test_strat))
M4.fit(X_train_cluster, y_train_cluster)
y_pred_cluster = M4.predict(X_test_cluster)
print("M4 Accuracy (Cluster Sampling):", M4.score(X_test_cluster,y_test_cluster))
M4.fit(X_train_random, y_train_random)
y_pred_random = M4.predict(X_test_random)
print("M4 Accuracy (Simple Random Sampling):", M4.score(X_test_random,y_test_random))



# M5
M5.fit(X_train_sys, y_train_sys)
y_pred_sys = M5.predict(X_test_sys)
print("M5 Accuracy (Systematic Sampling):", M5.score(X_test_sys,y_test_sys))
M5.fit(X_train_strat, y_train_strat)
y_pred_strat = M5.predict(X_test_strat)
print("M5 Accuracy (Stratified Sampling):", M5.score(X_test_strat,y_test_strat))
M5.fit(X_train_cluster, y_train_cluster)
y_pred_cluster = M5.predict(X_test_cluster)
print("M5 Accuracy (Cluster Sampling):", M5.score(X_test_cluster,y_test_cluster))
M5.fit(X_train_random, y_train_random)
y_pred_random = M5.predict(X_test_random)
print("M5 Accuracy (Simple Random Sampling):", M5.score(X_test_random,y_test_random))



