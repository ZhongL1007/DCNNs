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
soil_property = 4208#The number of columns where the soil_property is located

data = df.iloc[:,1:soil_property].values
txt = open('LucasResNet_16.txt','w')#Create a new .txt file to save accuracy and other information 
soil_property_max = np.amax(data[:,soil_property-2])
print(f"soil_property_max:{soil_property_max}",file=txt)

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
train_soil_property = train[:,soil_property-2]
train_soil_property_normalization = np.divide(train_soil_property,soil_property_max)
test_soil_property = test[:,soil_property-2]
test_soil_property_normalization = np.divide(test_soil_property,soil_property_max)

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

def LucasResNet_16():
    inpt = Input(shape=input_shape)
    
    x = Conv2d_BN(inpt,nb_filter=6,kernel_size=(1,7),strides=(1,2),padding='same')
    x = MaxPooling2D(pool_size=(1,3),strides=(1,2),padding='same')(x)
    
    x = Conv_Block(x,nb_filter=[6,6,12],kernel_size=(1,3),strides=(1,1),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=[6,6,12],kernel_size=(1,3),strides=(1,1))

    x = Conv_Block(x,nb_filter=[12,12,24],kernel_size=(1,3),strides=(1,2),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=[12,12,24],kernel_size=(1,3),strides=(1,1))
    
    x = MaxPooling2D(pool_size=(1,3),strides=(1,2))(x)
    x = Flatten()(x)
    
    x = Dense(200,activation='tanh')(x)
    x = Dropout(0.3)(x)
    x = Dense(100,activation='tanh')(x)
    x = Dropout(0.3)(x)
    x = Dense(1,activation='sigmoid')(x)    
    
    model = Model(inputs=inpt,outputs=x)
    return model

model = LucasResNet_16()
model.summary()

#Setting parameters, compile and fit
nadam = optimizers.Nadam(lr=0.0001, epsilon=1e-08)
early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=2)
model.compile(optimizer=nadam,loss='mse')
hist = model.fit(train_x,train_soil_property_normalization,epochs=10000,batch_size=32,validation_split=0.2,verbose=2, shuffle=True, callbacks=[early_stopping])

#predict and save
predict = model.predict(test_x)*soil_property_max
predict_values = np.squeeze(predict)
pre=pd.DataFrame(data=predict_values)
pre.to_csv('LucasResNet_16_predict_values.csv')

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
R2_soil_property = R2(test_soil_property, predict_values)
RMSE_soil_property = RMSE(test_soil_property, predict_values)
RPD_soil_property = RPD(test_soil_property, predict_values)
print(f"Test_R2_soil_property:{R2_soil_property}",file=txt)
print(f"RMSE_soil_property:{RMSE_soil_property}",file=txt)
print(f"RPD_soil_property:{RPD_soil_property}",file=txt)

#Graph of loss change during training
plt.figure()
plt.plot(np.arange(0, len(hist.history["loss"])), hist.history["loss"], label="Calibration")
plt.plot(np.arange(0, len(hist.history["loss"])), hist.history["val_loss"], label="Validation")
plt.title("Calibration/Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.savefig("LucasResNet_16_loss.jpg",dpi=300)

#Save the model
model.save('LucasResNet_16.h5')

#Running time
end = time.time()
print (time.strftime(f"%Y-%m-%d %H:%M:%S,{time.localtime()}"),file=txt)
print("Running time:%.2f秒"%(end-start),file=txt)
txt.close()
