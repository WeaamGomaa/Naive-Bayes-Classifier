import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

import CategoricalNaiveBayes
import PCA

df = pd.read_csv("../NaiveBayes_Assignment/mushrooms.csv")
df.head()

#Handle missing values
df['stalk-root'] = df['stalk-root'].replace('?', 'unknown')

#Separate features and target
X = df.drop(columns=['class'])
Y = df['class']

# Features Encoding:
oe = OrdinalEncoder()
X_encoded = oe.fit_transform(X)

X_encoded = pd.DataFrame(X_encoded, columns=X.columns)

le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

# Verify
print("X_encoded shape:", X_encoded.shape)
print("y_encoded unique values:", np.unique(Y_encoded))
print(X_encoded.head())

#____________________________________________________________________________

#Splitting Dataset:
X_temp, X_test, Y_temp, Y_test = train_test_split(X_encoded, Y_encoded, test_size=0.2, random_state=42)

X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.25, random_state=42)

print(f"Train size:      {X_train.shape}")
print(f"Validation size: {X_val.shape}")
print(f"Test size:       {X_test.shape}")

#____________________________________________________________________________

#Scaling:
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled   = scaler.transform(X_val)
# X_test_scaled  = scaler.transform(X_test)
# print(X_train_scaled)
# print(X_val_scaled)
# print(X_test_scaled)
#____________________________________________________________________________

# Calculate the Accuracy:
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

#____________________________________________________________________________

#Experiment 0:(All Features + Naive Bayes)
naive_bayes = CategoricalNaiveBayes.CategoricalNaiveBayes()
naive_bayes.fit(X_train, Y_train)
exper0_predictions = naive_bayes.predict(X_test)

print("Experiment 0 Accuracy:", accuracy(Y_test, exper0_predictions))

#_____________________________________________________________________________

#Experiment B: PCA (Feature Reduction) + Naive Bayes:
k_values = [2, 5, 10, 15, 18, 21]
results = []

for k in k_values:
    pca = PCA.PCA(k)
    pca.fit(np.array(X_train))
    X_test_pca = pca.transform(np.array(X_test))

    experB_preds = naive_bayes.predict(X_test_pca)
    acc = accuracy(Y_test, experB_preds)
    results.append((k, acc))
    print(f"k={k}: Accuracy = {acc:.4f}")


