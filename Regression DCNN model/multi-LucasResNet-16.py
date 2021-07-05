from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Input, add
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#Read data
start = time.time()
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
df=pd.read_csv(r'LUCAS_Soil_Dataset.csv')#Read data that has been randomly shuffled
x_num = 4200#Number of wavelengths
y_max = 4213#Maximum number of columns required to read
pH = 4207#The number of columns where the pH property is located 
OC = 4208#The number of columns where the OC property is located
CaCO3 = 4209#The number of columns where the CaCO3 property is located
N = 4210#The number of columns where the N property is located
P = 4211#The number of columns where the P property is located
K = 4212#The number of columns where the K property is located
CEC = 4213#The number of columns where the CEC property is located

data = df.iloc[:,1:y_max].values
txt = open('multi_LucasResNet_16.txt','w')#Create a new .txt file to save accuracy and other information 
pH_max = np.amax(data[:,pH-2])
OC_max = np.amax(data[:,OC-2])
CaCO3_max = np.amax(data[:,CaCO3-2])
N_max = np.amax(data[:,N-2])
P_max = np.amax(data[:,P-2])
K_max = np.amax(data[:,K-2])
CEC_max = np.amax(data[:,CEC-2])
print(f"pH_max:{pH_max}",file=txt)
print(f"OC_max:{OC_max}",file=txt)
print(f"CaCO3_max:{CaCO3_max}",file=txt)
print(f"N_max:{N_max}",file=txt)
print(f"P_max:{P_max}",file=txt)
print(f"K_max:{K_max}",file=txt)
print(f"CEC_max:{CEC_max}",file=txt)

#Divide training set and testing set
ratio = int(0.75*len(data))
train = data[:ratio]
test = data[ratio:]
train_x = train[:,:x_num]
train_x = train_x.reshape(train_x.shape[0], 1, x_num, 1)    
test_x = test[:,:x_num]
test_x = test_x.reshape(test_x.shape[0], 1, x_num, 1)
print(train_x.shape,file=txt)
print(test_x.shape,file=txt)

#Normalized soil properties values    
train_pH = train[:,pH-2]
train_pH_normalization = np.divide(train_pH,pH_max)
train_OC = train[:,OC-2]
train_OC_normalization = np.divide(train_OC,OC_max)
train_CaCO3 = train[:,CaCO3-2]
train_CaCO3_normalization = np.divide(train_CaCO3,CaCO3_max)
train_N = train[:,N-2]
train_N_normalization = np.divide(train_N,N_max)
train_P = train[:,P-2]
train_P_normalization = np.divide(train_P,P_max)
train_K = train[:,K-2]
train_K_normalization = np.divide(train_K,K_max)
train_CEC = train[:,CEC-2]
train_CEC_normalization = np.divide(train_CEC,CEC_max)

test_pH = test[:,pH-2]
test_pH_normalization = np.divide(test_pH,pH_max)
test_OC = test[:,OC-2]
test_OC_normalization = np.divide(test_OC,OC_max)
test_CaCO3 = test[:,CaCO3-2]
test_CaCO3_normalization = np.divide(test_CaCO3,CaCO3_max)
test_N = test[:,N-2]
test_N_normalization = np.divide(test_N,N_max)
test_P = test[:,P-2]
test_P_normalization = np.divide(test_P,P_max)
test_K = test[:,K-2]
test_K_normalization = np.divide(test_K,K_max)
test_CEC = test[:,CEC-2]
test_CEC_normalization = np.divide(test_CEC,CEC_max)

input_shape = (1, x_num, 1)#input_shape = (1, 4200, 1)

#Build model 
def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):
    if name is not None:
        conv_name = name + '_conv'
    else:
        conv_name = None

    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='tanh',name=conv_name)(x)
    return x

def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt,nb_filter=nb_filter[0],kernel_size=(1,1),strides=strides,padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(1,3), padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1,1), padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter[2],strides=strides,kernel_size=kernel_size)
        x = add([x,shortcut])
        return x
    else:
        x = add([x,inpt])
        return x

def multi_LucasResNet_16():
    inpt = Input(shape=input_shape)
    
    x = Conv2d_BN(inpt,nb_filter=6,kernel_size=(1,7),strides=(1,2),padding='same')
    x = MaxPooling2D(pool_size=(1,3),strides=(1,2),padding='same')(x)
    
    x = Conv_Block(x,nb_filter=[6,6,12],kernel_size=(1,3),strides=(1,1),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=[6,6,12],kernel_size=(1,3),strides=(1,1))

    x = Conv_Block(x,nb_filter=[12,12,24],kernel_size=(1,3),strides=(1,2),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=[12,12,24],kernel_size=(1,3),strides=(1,1))
    
    x = MaxPooling2D(pool_size=(1,3),strides=(1,2))(x)
    x_share = Flatten()(x)
    
    x_pH = Dense(200,activation='tanh')(x_share)
    x_pH = Dropout(0.3)(x_pH)
    x_pH = Dense(100,activation='tanh')(x_pH)
    x_pH = Dropout(0.3)(x_pH)
    x_pH = Dense(1,activation='sigmoid',name = 'pH_output')(x_pH)    
    
    x_OC = Dense(200,activation='tanh')(x_share)
    x_OC = Dropout(0.3)(x_OC)
    x_OC = Dense(100,activation='tanh')(x_OC)
    x_OC = Dropout(0.3)(x_OC)
    x_OC = Dense(1,activation='sigmoid',name = 'OC_output')(x_OC)

    x_CaCO3 = Dense(200,activation='tanh')(x_share)
    x_CaCO3 = Dropout(0.3)(x_CaCO3)
    x_CaCO3 = Dense(100,activation='tanh')(x_CaCO3)
    x_CaCO3 = Dropout(0.3)(x_CaCO3)
    x_CaCO3 = Dense(1,activation='sigmoid',name = 'CaCO3_output')(x_CaCO3)
    
    x_N = Dense(200,activation='tanh')(x_share)
    x_N = Dropout(0.3)(x_N)
    x_N = Dense(100,activation='tanh')(x_N)
    x_N = Dropout(0.3)(x_N)
    x_N = Dense(1,activation='sigmoid',name = 'N_output')(x_N)
    
    x_P = Dense(200,activation='tanh')(x_share)
    x_P = Dropout(0.3)(x_P)
    x_P = Dense(100,activation='tanh')(x_P)
    x_P = Dropout(0.3)(x_P)
    x_P = Dense(1,activation='sigmoid',name = 'P_output')(x_P)
    
    x_K = Dense(200,activation='tanh')(x_share)
    x_K = Dropout(0.3)(x_K)
    x_K = Dense(100,activation='tanh')(x_K)
    x_K = Dropout(0.3)(x_K)
    x_K = Dense(1,activation='sigmoid',name = 'K_output')(x_K)    
    
    x_CEC = Dense(200,activation='tanh')(x_share)
    x_CEC = Dropout(0.3)(x_CEC)
    x_CEC = Dense(100,activation='tanh')(x_CEC)
    x_CEC = Dropout(0.3)(x_CEC)
    x_CEC = Dense(1,activation='sigmoid',name = 'CEC_output')(x_CEC)
    
    model = Model(inputs=inpt,outputs=[x_pH,x_OC,x_CaCO3,x_N,x_P,x_K,x_CEC])
    return model

model = multi_LucasResNet_16()
model.summary()

#Setting parameters, compile and fit
nadam = optimizers.Nadam(lr=0.0001, epsilon=1e-08)
early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=2)
model.compile(optimizer=nadam,loss={'pH_output':'mse','OC_output':'mse','CaCO3_output':'mse','N_output':'mse','P_output':'mse','K_output':'mse','CEC_output':'mse'})
hist = model.fit(train_x,[train_pH_normalization,train_OC_normalization,train_CaCO3_normalization,train_N_normalization,train_P_normalization,train_K_normalization,train_CEC_normalization],
                 epochs=10000,batch_size=32,validation_split=0.2,verbose=2, shuffle=True, callbacks=[early_stopping])

#predict and save
predict = model.predict(test_x)
predict_values = np.squeeze(predict)
predict_pH = predict_values[0]*pH_max
predict_OC = predict_values[1]*OC_max
predict_CaCO3 = predict_values[2]*CaCO3_max
predict_N = predict_values[3]*N_max
predict_P = predict_values[4]*P_max
predict_K = predict_values[5]*K_max
predict_CEC = predict_values[6]*CEC_max
predict_all=np.vstack((predict_pH,predict_OC,predict_CaCO3,predict_N,predict_P,predict_K,predict_CEC))
predict_all_transpose = np.transpose(predict_all)
pre=pd.DataFrame(data=predict_all_transpose)
pre.to_csv('multi_LucasResNet_16_predict_values.csv')

#Define R2, RMSE, RPD
def R2(measured_values, predicted_values): 
    SS_res = sum((measured_values-predicted_values )**2)
    SS_tot = sum((measured_values - np.mean(measured_values))**2)
    return 1 - SS_res/SS_tot
def RMSE(measured_values, predicted_values):
    return (np.mean((predicted_values - measured_values)**2))**0.5
def RPD(measured_values, predicted_values):
    return ((np.mean((measured_values-np.mean(measured_values))**2))**0.5)/((np.mean((predicted_values - measured_values)**2))**0.5)

#Test accuracy
R2_pH = R2(test_pH, predict_pH)
RMSE_pH = RMSE(test_pH, predict_pH)
RPD_pH = RPD(test_pH, predict_pH)
print(f"Test_R2_pH:{R2_pH}",file=txt)
print(f"RMSE_pH:{RMSE_pH}",file=txt)
print(f"RPD_pH:{RPD_pH}",file=txt)

R2_OC = R2(test_OC, predict_OC)
RMSE_OC = RMSE(test_OC, predict_OC)
RPD_OC = RPD(test_OC, predict_OC)
print(f"Test_R2_OC:{R2_OC}",file=txt)
print(f"RMSE_OC:{RMSE_OC}",file=txt)
print(f"RPD_OC:{RPD_OC}",file=txt)

R2_CaCO3 = R2(test_CaCO3, predict_CaCO3)
RMSE_CaCO3 = RMSE(test_CaCO3, predict_CaCO3)
RPD_CaCO3 = RPD(test_CaCO3, predict_CaCO3)
print(f"Test_R2_CaCO3:{R2_CaCO3}",file=txt)
print(f"RMSE_CaCO3:{RMSE_CaCO3}",file=txt)
print(f"RPD_CaCO3:{RPD_CaCO3}",file=txt)

R2_N = R2(test_N, predict_N)
RMSE_N = RMSE(test_N, predict_N)
RPD_N = RPD(test_N, predict_N)
print(f"Test_R2_N:{R2_N}",file=txt)
print(f"RMSE_N:{RMSE_N}",file=txt)
print(f"RPD_N:{RPD_N}",file=txt)

R2_P = R2(test_P, predict_P)
RMSE_P = RMSE(test_P, predict_P)
RPD_P = RPD(test_P, predict_P)
print(f"Test_R2_P:{R2_P}",file=txt)
print(f"RMSE_P:{RMSE_P}",file=txt)
print(f"RPD_P:{RPD_P}",file=txt)

R2_K = R2(test_K, predict_K)
RMSE_K = RMSE(test_K, predict_K)
RPD_K = RPD(test_K, predict_K)
print(f"Test_R2_K:{R2_K}",file=txt)
print(f"RMSE_K:{RMSE_K}",file=txt)
print(f"RPD_K:{RPD_K}",file=txt)

R2_CEC = R2(test_CEC, predict_CEC)
RMSE_CEC = RMSE(test_CEC, predict_CEC)
RPD_CEC = RPD(test_CEC, predict_CEC)
print(f"Test_R2_CEC:{R2_CEC}",file=txt)
print(f"RMSE_CEC:{RMSE_CEC}",file=txt)
print(f"RPD_CEC:{RPD_CEC}",file=txt)

#Graph of loss change during training
plt.figure()
plt.plot(np.arange(0, len(hist.history["loss"])), hist.history["loss"], label="Calibration")
plt.plot(np.arange(0, len(hist.history["loss"])), hist.history["val_loss"], label="Validation")
plt.title("Calibration/Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.savefig("multi_LucasResNet_16_loss.jpg",dpi=300)

#Save the model
model.save('multi_LucasResNet_16.h5')

#Running time
end = time.time()
print (time.strftime(f"%Y-%m-%d %H:%M:%S,{time.localtime()}"),file=txt)
print("Running time:%.2f秒"%(end-start),file=txt)
txt.close()
