import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

X = np.loadtxt("X.csv", delimiter=" ")
y = np.loadtxt("y.csv", delimiter=" ")
X_val = np.loadtxt("X_val.csv", delimiter=" ")
y_val = np.loadtxt("y_val.csv", delimiter=" ")

# Arboles de desición

clf = tree.DecisionTreeClassifier(criterion="entropy")
clf.fit(X, y)

y_pred = clf.predict(X)
resultado_entrenamiento = np.mean(y_pred == y)

y_val_pred = clf.predict(X_val)
resultado_validacion = np.mean(y_val_pred == y_val)
cm = confusion_matrix(y_val, y_val_pred)

fig , ax = plt.subplots(figsize = (2.5, 2.5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax = ax)
ax.set_xlabel("Predicción")
ax.set_ylabel("Verdadaero")
plt.show()