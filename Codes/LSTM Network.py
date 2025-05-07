"""**Imported Libraries:**"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import activations
from tensorflow.keras.callbacks import ModelCheckpoint
from pyparsing.helpers import printables
from matplotlib import pyplot as plt
import seaborn as sns

"""**Read CSV Files and Delete NaN/empty Columns and Rows:**"""

Train_csv = pd.read_csv("Train.csv").dropna(axis="columns",how='all').dropna(axis="rows",how='all')
Val_csv  = pd.read_csv("Val.csv").dropna(axis="columns",how='all').dropna(axis="rows",how='all')
Test_csv = pd.read_csv("Test.csv").dropna(axis="columns",how='all').dropna(axis="rows",how='all')

"""**One hot Encoding:**"""

Cf_ohe = OneHotEncoder()
Ct_ohe = OneHotEncoder()

Cf_train_encoded = Cf_ohe.fit_transform(Train_csv[['C_f']]).toarray()
Ct_train_encoded = Ct_ohe.fit_transform(Train_csv[['C_theta']]).toarray()

Cf_val_encoded = Cf_ohe.transform(Val_csv[['C_f']]).toarray()
Ct_val_encoded = Ct_ohe.transform(Val_csv[['C_theta']]).toarray()

Cf_test_encoded = Cf_ohe.transform(Test_csv[['C_f']]).toarray()
Ct_test_encoded = Ct_ohe.transform(Test_csv[['C_theta']]).toarray()

"""**Normalize the data:**

"""

#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized

#Set the scale to which the values should be normalized:
scaler = MinMaxScaler(feature_range=(0.05, 0.95))

#scale all the data based on the distribution of the training data:
train_data_scaled = scaler.fit_transform(Train_csv[['X_v', 'Y_v', 'Theta_v', 'X_a', 'Y_a','Theta_a']])
val_data_scaled = scaler.transform(Val_csv[['X_v', 'Y_v', 'Theta_v', 'X_a', 'Y_a','Theta_a']])
test_data_scaled = scaler.transform(Test_csv[['X_v', 'Y_v', 'Theta_v', 'X_a', 'Y_a','Theta_a']])

print(train_data_scaled.shape)
print(val_data_scaled.shape)
print(test_data_scaled.shape)

#stack the encoded commands with the scaled data

train_data_scaled = np.hstack((Cf_train_encoded, Ct_train_encoded,train_data_scaled))
val_data_scaled = np.hstack((Cf_val_encoded, Ct_val_encoded,val_data_scaled))
test_data_scaled = np.hstack((Cf_test_encoded, Ct_test_encoded,test_data_scaled))


print(train_data_scaled.shape)
print(val_data_scaled.shape)
print(test_data_scaled.shape)

"""**Time-Series Data:**"""

#Empty lists to be populated using formatted training data
trainX = []
trainY = []

valX = []
valY = []

testX = []
testY = []

Seq_length = 50  #Input Sequence Length

#Converts data to time-series and jumps with the sequence length once the sequence ends to avoid overlapping

i = Seq_length
while i < len(train_data_scaled):
  trainX.append(train_data_scaled[i - Seq_length:i, 0:17])
  trainY.append(train_data_scaled[i, 17:20])
  i = i+1
  if i != len(train_data_scaled):
    if Train_csv.at[Train_csv.index[i],'time'] == 0.0:
      print(i)
      i = i + Seq_length

i = Seq_length
val_split = 0
while i < len(val_data_scaled):
    valX.append(val_data_scaled[i - Seq_length:i, 0:17])
    valY.append(val_data_scaled[i, 17:20])
    i = i+1
    if i != len(val_data_scaled):
      if Val_csv.at[Val_csv.index[i],'time'] == 0.0:
        val_split = i
        print(val_split)
        i = i + Seq_length

i = Seq_length
test_split= 0
while i < len(test_data_scaled):
    testX.append(test_data_scaled[i - Seq_length:i, 0:17])
    testY.append(test_data_scaled[i, 17:20])
    i = i+1
    if i != len(test_data_scaled):
     if Test_csv.at[Test_csv.index[i],'time'] == 0.0:
        test_split = i
        print(test_split)
        i = i + Seq_length

trainX= np.array(trainX)   #convert from tuples to numpy array
trainY = np.array(trainY)
print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

valX= np.array(valX)
valY = np.array(valY)
print('valX shape == {}.'.format(valX.shape))
print('valY shape == {}.'.format(valY.shape))

testX= np.array(testX)
testY = np.array(testY)
print('testX shape == {}.'.format(testX.shape))
print('testY shape == {}.'.format(testY.shape))
print(trainX[1][1])

"""**LSTM Model:**"""

model = Sequential()
model.add(LSTM(256, activation='tanh', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(256, activation='tanh', return_sequences=False))
model.add(Dropout(0.4))
model.add(Dense(256, activation= 'relu'))
model.add(Dense(trainY.shape[1], activation= 'sigmoid'))
model.compile(optimizer= tensorflow.keras.optimizers.Adam(learning_rate=0.001) , loss='mae')
model.summary()

save_best_model = ModelCheckpoint("mdl_wts.hdf5", monitor='val_loss', save_best_only=True, save_weights_only=True)

"""**Training:**"""

#An optimal batch-size is 64.
#There might be some cases wherethe batch size is 32, 64, 128 which must be dividable by 8.

history = model.fit(trainX, trainY, epochs=300,validation_data=(valX, valY), batch_size = 32, callbacks=[save_best_model])

"""**Loss Curves Plotting:**"""

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

"""**Testing:**"""

#Returns the loss value & metrics values for the model in test mode.
model.load_weights("mdl_wts.hdf5")

RMSE_test = model.evaluate(testX, testY)
print(RMSE_test)

#Denormalization: (the normalizing function originally had 6 columns, we need the 3 final columns only)
test_prediction = model.predict(testX)

prediction_copies = np.concatenate((test_prediction,test_prediction), axis=1)
test_prediction_denormalized = scaler.inverse_transform(prediction_copies)[:,3:6]
print(test_prediction_denormalized)

#create the predicted output CSV file (take the missing readings from the dynamic model)
test_evaluation_csv= pd.DataFrame(test_prediction_denormalized, columns=['X_v', 'Y_v','Theta_v'])

print(test_evaluation_csv.shape, Test_csv.shape)
test_seq1 = pd.concat([Test_csv[['X_v','Y_v','Theta_v']][0:Seq_length],  test_evaluation_csv[:][0:test_split-Seq_length]] , ignore_index=True , axis=0)
test_seq2 = pd.concat([Test_csv[['X_v','Y_v','Theta_v']][test_split:test_split+Seq_length],  test_evaluation_csv[:][test_split-Seq_length:]] , ignore_index=True , axis=0)


print(test_evaluation_csv.shape, Test_csv.shape)
test_seq1.to_csv("test1.csv")
test_seq2.to_csv("test2.csv")

# Calculating the ATE and Theta error (RMSE)

Test_euclidean1 =((Test_csv['X_a'][0:test_split].values-test_seq1['X_v'].values)**2 + (Test_csv['Y_a'][0:test_split].values - test_seq1['Y_v'].values)**2)**0.5
Test_euclidean2 =((Test_csv['X_a'][test_split:].values-test_seq2['X_v'].values)**2 + (Test_csv['Y_a'][test_split:].values - test_seq2['Y_v'].values)**2)**0.5

print(Test_euclidean1.shape,Test_euclidean2.shape )
ATE_test_1 = (Test_euclidean1**2).mean()**0.5
ATE_test_2 = (Test_euclidean2**2).mean()**0.5

print("ATE of Sequence1 = " , ATE_test_1)
print("ATE of Sequence2 = " , ATE_test_2)

#-----------------------------------------------------

def Theta_modify(x):
  if x >= 180:
    return 360-x
  else:
    return x

theta_update = np.vectorize(Theta_modify)

Theta_diff1 = abs(Test_csv['Theta_a'][0:test_split].values-test_seq1['Theta_v'].values)
Theta_diff1 = theta_update(Theta_diff1)
Theta_diff2 = abs(Test_csv['Theta_a'][test_split:].values-test_seq2['Theta_v'].values)
Theta_diff2 = theta_update(Theta_diff2)

ThetaError_test_1 = (Theta_diff1**2).mean()**0.5
ThetaError_test_2 = (Theta_diff2**2).mean()**0.5

print("Theta Error of Sequence1 = " , ThetaError_test_1)
print("Theta Error Sequence2 = " , ThetaError_test_2)

"""**Validation:**"""

#Returns the loss value & metrics values for the model in test mode.

RMSE_val = model.evaluate(valX, valY)
print(RMSE_val)

#Denormalization: (the normalizing function originally had 6 columns, we need the 3 final columns only)
val_prediction = model.predict(valX)

prediction_copies = np.concatenate((val_prediction,val_prediction), axis=1)

val_prediction_denormalized = scaler.inverse_transform(prediction_copies)[:,3:6]

#create the predicted output CSV file (take the missing readings from the dynamic model)

val_evaluation_csv= pd.DataFrame(val_prediction_denormalized, columns=['X_v', 'Y_v','Theta_v'])

print(val_evaluation_csv.shape, Val_csv.shape)
val_seq1 = pd.concat([Val_csv[['X_v','Y_v','Theta_v']][0:Seq_length],  val_evaluation_csv[:][0:val_split-Seq_length]] , ignore_index=True , axis=0)
val_seq2 = pd.concat([Val_csv[['X_v','Y_v','Theta_v']][val_split:val_split+Seq_length],  val_evaluation_csv[:][val_split-Seq_length:]] , ignore_index=True , axis=0)


print(val_evaluation_csv.shape, Val_csv.shape)
val_seq1.to_csv("val1.csv")
val_seq2.to_csv("val2.csv")

# Calculating the ATE and Theta error (RMSE)

Val_euclidean1 =((Val_csv['X_a'][0:val_split].values-val_seq1['X_v'].values)**2 + (Val_csv['Y_a'][0:val_split].values - val_seq1['Y_v'].values)**2)**0.5
Val_euclidean2 =((Val_csv['X_a'][val_split:].values-val_seq2['X_v'].values)**2 + (Val_csv['Y_a'][val_split:].values - val_seq2['Y_v'].values)**2)**0.5

print(Val_euclidean1.shape,Val_euclidean2.shape )
ATE_val_1 = (Val_euclidean1**2).mean()**0.5
ATE_val_2 = (Val_euclidean2**2).mean()**0.5

print("ATE of Sequence1 = " , ATE_val_1)
print("ATE of Sequence2 = " , ATE_val_2)

#------------------------------------------------------

def Theta_modify(x):
  if x >= 180:
    return 360-x
  else:
    return x

theta_update = np.vectorize(Theta_modify)

Theta_diff1 = abs(Val_csv['Theta_a'][0:val_split].values-val_seq1['Theta_v'].values)
Theta_diff1 = theta_update(Theta_diff1)
Theta_diff2 = abs(Val_csv['Theta_a'][val_split:].values-val_seq2['Theta_v'].values)
Theta_diff2 = theta_update(Theta_diff2)

ThetaError_val_1 = (Theta_diff1**2).mean()**0.5
ThetaError_val_2 = (Theta_diff2**2).mean()**0.5

print("Theta Error of Sequence1 = " , ThetaError_val_1)
print("Theta Error Sequence2 = " , ThetaError_val_2)