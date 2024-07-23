import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import os
from keras.layers import Dropout, Activation
import snf
from keras.layers import Input,Dense,Add
from sklearn.linear_model import LogisticRegression
from keras.layers import Flatten, Conv2D, MaxPooling2D
 
# Load the interaction matrix and similarity matrices
interaction_matrix = pd.read_csv('Main_Dataset/Jac_Training_Interaction.csv', index_col=0)
sim_matrix_transporter = pd.read_csv('Main_Dataset/Transporter_Sim_Interaction.csv', index_col=0)
sim_matrix_enzyme = pd.read_csv('Main_Dataset/Sim_Enzyme_Interaction.csv', index_col=0)
sim_matrix_offside = pd.read_csv('Main_Dataset/Off_Sim_Interaction.csv', index_col=0)
sim_matrix_carrier = pd.read_csv('Main_Dataset/Carrier_Sim_Interaction.csv', index_col=0)
sim_matrix_target = pd.read_csv('Main_Dataset/Target_Sim_Interaction.csv', index_col=0)
sim_matrix_sideeffect = pd.read_csv('Main_Dataset/SE_Sim_Interaction.csv', index_col=0)
sim_matrix_chem = pd.read_csv('Main_Dataset/Chemsub_Sim_Interaction.csv', index_col=0)
interaction = np.loadtxt("Main_Dataset/Jac_Training_Interaction.csv",dtype=float,delimiter=",")
 
# Define the similarity matrices and their names
sim_dict = {
    'transporter': sim_matrix_transporter,
    'enzyme': sim_matrix_enzyme,
    'offside': sim_matrix_offside,
    'carrier': sim_matrix_carrier,
    'target': sim_matrix_target,
    'sideeffect': sim_matrix_sideeffect,
    'chem': sim_matrix_chem
}
 
# Extract names and matrices
names_list = list(sim_dict.keys())
matrices_list = list(sim_dict.values())
 
# Generate all combinations from single matrices to all matrices
def generate_combinations(names_list):
    for length in range(2, len(names_list) + 1):
        for combo in itertools.combinations(names_list, length):
            yield combo
 
# Function to prepare features and labels
def prepare_data(fused_network):
    drug_fea = fused_network
    train = []
    label = []
    tmp_fea=[]
    drug_fea_tmp = []
    total_sum = 0
    count = 0
    for i in range(200):
        for j in range(i+1,200):
            label.append(interaction[i,j])
            drug_fea_tmp = list(drug_fea[i])
            drug_fea_tmp2 = list(drug_fea[j])
            tmp_fea = (drug_fea_tmp+drug_fea_tmp2)
            train.append(tmp_fea)
    return np.array(train), label
 
# Store accuracies
accuracies = {}
 
start = 0
end = 20

# Iterate through all combinations, train and evaluate models
for name_combo in generate_combinations(names_list):
    if start>=end:
        break
    combined_matrices = [sim_dict[name] for name in name_combo]
    fused_network = snf.snf(combined_matrices, K=40)
    # Prepare data
    X,y = prepare_data(fused_network)
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)
    
    # Split data
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1], 1))
    X_val= np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1], 1))
 
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    y_val = np.asarray(y_val)
    num_classes = 2
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    start+=1
    
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
 
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    # Train a model
    model.pop()
    features_train = model.predict(X_train)
    features_test = model.predict(X_test)
    y_train_1d = np.argmax(y_train, axis=1)
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(features_train, y_train_1d)
    y_pred = lr_model.predict(features_test)
    
    # Predict and evaluate
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store accuracy
    accuracies[name_combo] = accuracy
    print(f"Combination: {name_combo}, Accuracy: {accuracy:.4f}")
    accuracies_df = pd.DataFrame(list(accuracies.items()), columns=['Combination', 'Accuracy'])
    accuracies_df.to_csv("hi1.csv")
 
# Optionally, you can save `accuracies` to a file
accuracies_df = pd.DataFrame(list(accuracies.items()), columns=['Combination', 'Accuracy'])
accuracies_df.to_csv("hi1.csv")
# if os.path.exists(file_path):
    # accuracies_df.to_csv(file_path, mode='a', header=False, index=False)
# else:
    # accuracies_df.to_csv(file_path, mode='w', header=True, index=False)