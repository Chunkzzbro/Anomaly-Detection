import sys
import yaml
import warnings
warnings.filterwarnings("ignore")

import datetime
import numpy as np
import pandas as pd
import random
pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', None)

import seaborn as sns

import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras import regularizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, roc_auc_score, auc,
                              precision_score, recall_score, roc_curve, precision_recall_curve,
                              precision_recall_fscore_support)


df = pd.read_csv(sys.argv[1])
train_dataset = sys.argv[1]
df.shape
df.head()
max_debit = 0
max_credit = 0
# Check if the 'debit' column exists in the DataFrame
if 'Debit' in df.columns:
    # Find the maximum value in the 'debit' column
    max_debit = df['Debit'].max()
print(f"The maximum value in the 'Debit' column is: {max_debit}")
if 'Credit' in df.columns:
    # Find the maximum value in the 'debit' column
    max_credit = df['Credit'].max()
# to remove rows with nan vaules
print(f"The maximum value in the 'Credit' column is: {max_credit}")

df.dropna(axis=0, how='any', inplace=True)

df['Label']=0
df.reset_index(inplace=True, drop=True)


X_train = df
X_train = X_train.drop(['Label'], axis=1)
X_train = X_train.values


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)


# No of Neurons in each Layer [6,5,3,2,3,5,6]
input_dim = X_train.shape[1]

# adjust here based on input features
#encoding_dim = 5
encoding_dim = int(sys.argv[4])
#print('Input dim '+str(input_dim)+' encoding dim '+str(encoding_dim))

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh",activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="tanh")(encoder)
encoder = Dense(int(2), activation="tanh")(encoder)
decoder = Dense(int(encoding_dim/ 2), activation='tanh')(encoder)
decoder = Dense(int(encoding_dim), activation='tanh')(decoder)
decoder = Dense(input_dim, activation='tanh')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

#print(autoencoder.summary())

nb_epoch = 100
batch_size = 50
autoencoder.compile(optimizer='adam', loss='mse')

t_ini = datetime.datetime.now()
history = autoencoder.fit(X_train_scaled, X_train_scaled,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.1,
                        verbose=0
                        )

t_fin = datetime.datetime.now()
#print('Time to run the model: {} Sec.'.format((t_fin - t_ini).total_seconds()))


df_history = pd.DataFrame(history.history)

#--------------------- Test data---------------------------

count_anom = 0
count_use = 0
sum_deb = 0
avg_deb = 0
sum_cred = 0
avg_cred = 0

test_data = pd.read_csv(sys.argv[2])
test_data['Label']=0

test_argument = sys.argv[2]

X_tst = test_data 
y_tst  = X_tst['Label']
X_tst  = X_tst.drop(['Label'], axis=1)
test_data_temp = X_tst
X_tst  = X_tst.values

X_tst_scaled  = scaler.transform(X_tst)
predictions = autoencoder.predict(X_tst_scaled)

mse = np.mean(np.power(X_tst_scaled - predictions, 2), axis=1)
df_error = pd.DataFrame({'reconstruction_error': mse, 'Label': y_tst}, index=y_tst.index)

numerical_cols = df.columns.values
numerical_cols = np.delete(numerical_cols, [numerical_cols.size-1])

data_n = pd.DataFrame(X_tst_scaled, index= y_tst.index, columns=numerical_cols)

def compute_error_per_dim(point):

    initial_pt = np.array(data_n.loc[point,:]).reshape(1,len(df.columns)-1)  
    reconstrcuted_pt = autoencoder.predict(initial_pt)

    return abs(np.array(initial_pt  - reconstrcuted_pt)[0])

#average = df_error['reconstruction_error'].mean()
average = np.percentile(df_error['reconstruction_error'], 70)

#outliers = df_error.index[df_error.reconstruction_error > float(sys.argv[3])].tolist()
outliers = df_error.index[df_error.reconstruction_error > 1].tolist()


RE_per_dim = {}
for ind in outliers:
    RE_per_dim[ind] = compute_error_per_dim(ind)

#find the high error values in features
REhigh_per_dim = {}
for ind in outliers:
#    REhigh_per_dim[ind] = np.array([ i for (i, v) in enumerate(RE_per_dim[ind]) if v > av ])
    REhigh_per_dim[ind] = np.array([ i for i in RE_per_dim[ind] > 1 ])
    for i in RE_per_dim[ind]:
        if i>1:
            count_anom = count_anom+1  
    for i in range(3,5):
        if(RE_per_dim[ind][i]>1):
            count_use = count_use+1
            
    
            
            
              

RE_per_dim = pd.DataFrame(RE_per_dim, index= numerical_cols).T
REhigh_per_dim = pd.DataFrame(REhigh_per_dim, index= numerical_cols).T

selected_rows = test_data.iloc[outliers]

# Calculate the average of the 'Debit' and 'Credit' columns for the selected rows
average_debit = selected_rows[selected_rows['Debit'] > max_debit]['Debit'].mean()

# Calculate the average of 'Credit' values that are greater than max2
average_credit = selected_rows[selected_rows['Credit'] > max_credit]['Credit'].mean()

print(outliers)
print(RE_per_dim)
print(REhigh_per_dim)
print()
print(f"The average Debit for selected rows where Debit is: {average_debit}")
print(f"The average Credit for selected rows where Credit is: {average_credit}")
print("The number of rows containing anomalies",len(outliers))
print("Total number of anomalies",count_anom)
print("Total number of anomalies in debit and credit",count_use)

if(count_use>20):
    user_input = input("The number of anomalies in debit and credit columns in this data seem to be high. Do you want to retrain the train dataset with the values of debit and credit in the csv file? Y/N")
    if(user_input=='Y'):
        duplicated_rows = df.head(100).reset_index(drop=True)
        random_debits = [random.uniform(max_debit,average_debit)for _ in range(100)]
        random_credits = [random.uniform(max_credit,average_credit)for _ in range(100)]
        duplicated_rows['Debit'] = random_debits
        duplicated_rows['Credit'] = random_credits
        
        df = pd.concat([df,duplicated_rows],ignore_index = True)
        df = df.drop(columns=['Label'])
        df.to_csv(train_dataset,index=False)
        
        

        
        
        
        
'''
#----------------------dumping data to yaml----------

# mode=0 - Validate only
# mode=1 - All
mode = int(sys.argv[5])

dict = {}
dict['Anomalies'] = outliers
dict['Features_List'] = numerical_cols.tolist() 
dict['Reconstruction_Error'] = df_error.to_dict('dict')
#dict['Threshold'] = [float(sys.argv[3])]
dict['Feature_Wise_Reconstruction_Error'] = RE_per_dim.to_dict('dict')
dict['Feature_Wise_High_Error'] = REhigh_per_dim.to_dict('dict')


if mode == 1:
    dict['Training_Data'] = df.to_dict('dict')
    dict['Test_Data'] = test_data_temp.to_dict('dict')

#print(yaml.dump(dict, default_flow_style=False))
print(yaml.dump(dict))

#----------------------dumping data to yaml----------
'''