# -*- coding: utf-8 -*-
'''
!pip install snfpy
!pip install --upgrade keras
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import snf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, f1_score, recall_score, precision_score, roc_curve, auc, roc_auc_score, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from keras.callbacks import Callback



# Load the interaction matrix from CSV file
interaction_matrix = pd.read_csv('Main_Dataset/Jac_Training_Interaction.csv', index_col=0)
sim_matrix_transporter = pd.read_csv('Main_Dataset/Transporter_Sim_Interaction.csv', index_col=0)
sim_matrix_enzyme = pd.read_csv('Main_Dataset/Sim_Enzyme_Interaction.csv', index_col=0)
sim_matrix_offside = pd.read_csv('Main_Dataset/Off_Sim_Interaction.csv', index_col=0)
sim_matrix_carrier = pd.read_csv('Main_Dataset/Carrier_Sim_Interaction.csv', index_col=0)
sim_matrix_target = pd.read_csv('Main_Dataset/Target_Sim_Interaction.csv', index_col=0)
sim_matrix_sideeffect = pd.read_csv('Main_Dataset/SE_Sim_Interaction.csv', index_col=0)
sim_matrix_chem = pd.read_csv('Main_Dataset/Chemsub_Sim_Interaction.csv', index_col=0)

interaction = np.loadtxt("Main_Dataset/Jac_Training_Interaction.csv", dtype=float, delimiter=",")

sim_data = {
    'sim_matrix_transporter': sim_matrix_transporter,
    'sim_matrix_enzyme': sim_matrix_enzyme,
    'sim_matrix_offside': sim_matrix_offside,
    'sim_matrix_carrier': sim_matrix_carrier,
    'sim_matrix_target': sim_matrix_target,
    'sim_matrix_sideeffect': sim_matrix_sideeffect,
    'sim_matrix_chem': sim_matrix_chem
}

similarity_matrices = [sim_matrix_target, sim_matrix_enzyme, sim_matrix_transporter, sim_matrix_carrier]

fused_network = snf.snf(similarity_matrices, K=40)

def prepare_data():
    drug_fea = fused_network
    interaction = np.loadtxt("Main_Dataset/Jac_Training_Interaction.csv", dtype=float, delimiter=",")
    train = []
    label = []
    for i in range(0, 10):
        for j in range(i + 1, 10):
            label.append(interaction[i, j])
            drug_fea_tmp = list(drug_fea[i])
            drug_fea_tmp2 = list(drug_fea[j])
            train.append(drug_fea_tmp + drug_fea_tmp2)
    return np.array(train), label

X, y = prepare_data()

X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1], 1))

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
X_val = np.asarray(X_val)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
y_val = np.asarray(y_val)

num_classes = 2
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

# Custom callback to calculate MCC after each epoch
class MCCCallback(Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.train_mccs = []
        self.val_mccs = []

    def on_epoch_end(self, epoch, logs=None):
        y_train_pred = (self.model.predict(self.validation_data[0]) > 0.5).astype("int32")
        y_val_pred = (self.model.predict(self.validation_data[1]) > 0.5).astype("int32")
        y_train_true = np.argmax(self.validation_data[2], axis=1)
        y_val_true = np.argmax(self.validation_data[3], axis=1)
        y_train_pred = np.argmax(y_train_pred, axis=1)
        y_val_pred = np.argmax(y_val_pred, axis=1)
        train_mcc = matthews_corrcoef(y_train_true, y_train_pred)
        val_mcc = matthews_corrcoef(y_val_true, y_val_pred)
        self.train_mccs.append(train_mcc)
        self.val_mccs.append(val_mcc)
        print(f"Epoch {epoch+1}: train_mcc={train_mcc:.4f}, val_mcc={val_mcc:.4f}")


# CNN
input_shape = (1, 1578, 1)

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

# Train model with MCC callback
mcc_callback = MCCCallback(validation_data=(X_train, X_val, y_train, y_val))
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=128, verbose=1, epochs=20, callbacks=[mcc_callback])
model.pop()

# Extracting CNN features for Random Forest
cnn_features_train = model.predict(X_train)
cnn_features_test = model.predict(X_test)

# Random Forest
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30]}
rf = RandomForestClassifier()
clf_rf = GridSearchCV(rf, rf_params, cv=5)
y_train_1d = np.argmax(y_train, axis=1)
clf_rf.fit(cnn_features_train, y_train_1d)
print(f'Best parameters for Random Forest: {clf_rf.best_params_}')
y_pred = clf_rf.predict(cnn_features_test)

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
roc_display.figure_.set_size_inches(5, 5)
plt.plot([0, 1], [0, 1], color='g')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.savefig(f"Plots/{model_name}_roc_curve.png")
plt.show()

# Print confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion)

# Save MCC values to a text file
with open("mcc_values.txt", "w") as f:
    f.write("Train MCC values:\n")
    f.write("\n".join(map(str, mcc_callback.train_mccs)))
    f.write("\n\nValidation MCC values:\n")
    f.write("\n".join(map(str, mcc_callback.val_mccs)))

# Plot MCC values over epochs
plt.plot(mcc_callback.train_mccs, label='Train MCC')
plt.plot(mcc_callback.val_mccs, label='Validation MCC')
plt.xlabel('Epochs')
plt.ylabel('MCC')
plt.legend()
plt.title('MCC over Epochs')
plt.savefig(f"Plots/{model_name}_mcc_over_epochs.png")
plt.show()
