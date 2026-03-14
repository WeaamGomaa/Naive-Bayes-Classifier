import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.metrics import classification_report, confusion_matrix

import CategoricalNaiveBayes
import PCA

df = pd.read_csv("mushrooms.csv")
df.head()

# Handle missing values
df['stalk-root'] = df['stalk-root'].replace('?', 'unknown')

# Separate features and target
X = df.drop(columns=['class'])
Y = df['class']

# Features Encoding:
oe = OrdinalEncoder()
X_encoded = oe.fit_transform(X)

X_encoded = pd.DataFrame(X_encoded, columns=X.columns)

le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

#____________________________________________________________________________

#Splitting Dataset:
X_temp, X_test, Y_temp, Y_test = train_test_split(X_encoded, Y_encoded, test_size=0.2, random_state=42)

X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.25, random_state=42)

#____________________________________________________________________________

# Evaluating Model:
def evaluate_model(y_true, y_pred, experiment_name):
    print(f"\n{'==========================================='}")
    print(f"  {experiment_name}")
    print(f"{'==========================================='}")

    acc = np.sum(y_true == y_pred) / len(y_true)
    print(f"\nAccuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Edible', 'Poisonous'], zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Edible', 'Poisonous'],
                yticklabels=['Edible', 'Poisonous'])
    plt.title(f'Confusion Matrix — {experiment_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{experiment_name}.png')
    plt.close()

    return acc

#____________________________________________________________________________

#Experiment 0:(All Features + Naive Bayes)
nb_0 = CategoricalNaiveBayes.CategoricalNaiveBayes()
nb_0.fit(X_train, Y_train)
exper0_preds = nb_0.predict(X_test)

# print("Experiment 0 Accuracy:", accuracy(Y_test, exper0_preds))
evaluate_model(Y_test, exper0_preds, "Experiment 0")

#_____________________________________________________________________________

#Experiment A: Feature Selection + Naive Bayes:
feature_selector = SelectKBest(score_func=chi2, k=10)
feature_selector.fit(X_train, Y_train)
X_train_selected = feature_selector.transform(X_train)
X_test_selected = feature_selector.transform(X_test)

nb_A = CategoricalNaiveBayes.CategoricalNaiveBayes()
nb_A.fit(X_train_selected, Y_train)
experA_preds = nb_A.predict(X_test_selected)

# print("Experiment A Accuracy:", accuracy(Y_test, experA_preds))
evaluate_model(Y_test, experA_preds, "Experiment A")

#_____________________________________________________________________________

#Experiment B: PCA (Feature Reduction) + Naive Bayes:
k_values = [2, 5, 10, 15, 18, 21]
results = []
for k in k_values:
    pca = PCA.PCA(k)
    pca.fit(np.array(X_train))
    X_train_pca = pca.transform(np.array(X_train))
    X_test_pca = pca.transform(np.array(X_test))
    nb_B = CategoricalNaiveBayes.CategoricalNaiveBayes()
    nb_B.fit(X_train_pca, Y_train)
    experB_preds = nb_B.predict(X_test_pca)
    evaluate_model(Y_test, experB_preds, "Experiment B")



