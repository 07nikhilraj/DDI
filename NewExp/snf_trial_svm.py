# -*- coding: utf-8 -*-
'''
!pip install snfpy
!pip install --upgrade keras
'''
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
import snf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sklearn
from operator import itemgetter
from heapq import nlargest
from scipy.spatial.distance import pdist, squareform
# from keras.constraints import maxnorm
from keras.layers import Input,Dense,Add
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.layers import Dropout, Activation
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from keras import optimizers
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV

# Load the interaction matrix from CSV file
interaction_matrix = pd.read_csv('Main_Dataset/Jac_Training_Interaction.csv', index_col=0)
sim_matrix_transporter = pd.read_csv('Main_Dataset/Transporter_Sim_Interaction.csv', index_col=0)
sim_matrix_enzyme = pd.read_csv('Main_Dataset/Sim_Enzyme_Interaction.csv', index_col=0)
sim_matrix_offside = pd.read_csv('Main_Dataset/Off_Sim_Interaction.csv', index_col=0)
sim_matrix_carrier = pd.read_csv('Main_Dataset/Carrier_Sim_Interaction.csv', index_col=0)
sim_matrix_target = pd.read_csv('Main_Dataset/Target_Sim_Interaction.csv', index_col=0)
sim_matrix_sideeffect = pd.read_csv('Main_Dataset/SE_Sim_Interaction.csv', index_col=0)
sim_matrix_chem = pd.read_csv('Main_Dataset/Chemsub_Sim_Interaction.csv', index_col=0)

interaction = np.loadtxt("Main_Dataset/Jac_Training_Interaction.csv",dtype=float,delimiter=",")

sim_data = {
    'sim_matrix_transporter':sim_matrix_transporter,
    'sim_matrix_enzyme':sim_matrix_enzyme,
    'sim_matrix_offside':sim_matrix_offside,
    'sim_matrix_carrier':sim_matrix_carrier,
    'sim_matrix_target':sim_matrix_target,
    'sim_matrix_sideeffect':sim_matrix_sideeffect,
    'sim_matrix_chem':sim_matrix_chem
}

sim_list = [sim_matrix_transporter,sim_matrix_enzyme,sim_matrix_offside,sim_matrix_carrier,sim_matrix_target,sim_matrix_sideeffect,sim_matrix_chem]

"""*   Carrier
*   Transporter
*   Target
*   Enzyme
"""

similarity_matrices = [sim_matrix_target ,sim_matrix_enzyme,sim_matrix_transporter,sim_matrix_carrier]

fused_network = snf.snf(similarity_matrices, K=40)

def prepare_data():
    drug_fea = fused_network
    interaction = np.loadtxt("Main_Dataset/Jac_Training_Interaction.csv",dtype=float,delimiter=",")
    train = []
    label = []
    tmp_fea=[]
    drug_fea_tmp = []
    total_sum = 0
    count = 0
    for i in range(0, 600):
        for j in range(i+1, 600):
            total_sum += interaction[i,j]
            count+=1
            label.append(interaction[i,j])
            drug_fea_tmp = list(drug_fea[i])
            drug_fea_tmp2 = list(drug_fea[j])
            tmp_fea = (drug_fea_tmp+drug_fea_tmp2)
            train.append(tmp_fea)
    return np.array(train), label

X, y = prepare_data()

X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1], 1))
X_val= np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1], 1))

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
X_val = np.asarray(X_val)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
y_val = np.asarray(y_val)

num_classes = 2
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

# CNN
# Define input shape
input_shape = (1,1578,1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(1, 2), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(1, 2), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(1, 2), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=(1, 2), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1), strides=(1, 1)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train,validation_data=(X_val, y_val), batch_size = 128, verbose = 1,epochs = 20)
model.pop()
# Extracting CNN features for hybrid CNN
# cnn_feature_extractor = Sequential([model.layers[0], model.layers[1], model.layers[2],model.layers[3],model.layers[4],model.layers[5],model.layers[6],model.layers[7],model.layers[8],model.layers[9],model.layers[10],model.layers[11]])
cnn_features_train = model.predict(X_train)
cnn_features_test = model.predict(X_test)

svm_params = {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
svm = SVC()
clf_svm = GridSearchCV(svm, svm_params, cv=5)
y_train_1d = np.argmax(y_train, axis=1)
clf_svm.fit(cnn_features_train, y_train_1d)
print(f'Best parameters for SVM: {clf_svm.best_params_}')
y_pred = clf_svm.predict(cnn_features_test)
unique, counts = np.unique(y_pred, return_counts=True)
print(f'Predicted class distribution: {dict(zip(unique, counts))}')



# lr_model = LogisticRegression(max_iter=1000)
# lr_model.fit(cnn_features_train, y_train_1d)

# CNN+LR
# ensemble_model = VotingClassifier(estimators=[('lr', lr_model)], voting='soft', weights=[1])
# ensemble_model.fit(cnn_features_train, y_train_1d)

# Run this separately for each CNN and change the model to {model, ensemble_model} as per requirement


# Update the name of the model to avoid saving different plots under same model name before running the next cell
model_name = "CNN_SVM_100_2"

# Run this separately for each CNN and change the model to {model, ensemble_model} as per requirement
# y_pred = model.predict(X_test)
y_pred[y_pred<0.5]=0
y_pred[y_pred>=0.5]=1
# y_pred = np.argmax(y_pred, axis=1)

# Calculate metrics
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
specificity = recall_score(y_test, y_pred, pos_label=0)
accuracy = accuracy_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

# Print metrics
print("Metrics:")
print("Recall: %.2f%%" % (recall * 100.0))
print("Precision: %.2f%%" % (precision * 100.0))
print("Specificity: %.2f%%" % (specificity * 100.0))
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("AUC-ROC: %.2f%%" % (auc_roc * 100.0))
print("F1 Score: %.2f%%" % (f1 * 100.0))
print("MCC: %.2f%%" % (mcc * 100.0))


# Calculate and plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
roc_display.figure_.set_size_inches(5,5)
plt.plot([0, 1], [0, 1], color = 'g')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
filename = f"Plots/{model_name}_roc_curve.png"
plt.savefig(filename)
plt.show()

# Print confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion)
