from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#The LUCAS topsoil dataset URL：http://esdac.jrc.ec.europa.eu/
#Python version：3.7.3
#keras version：2.3.1

#Create a new .txt file to save accuracy and other information
start = time.time()
txt = open('LucasVGGNet_16.txt','w')
print (time.strftime(f"%Y-%m-%d %H:%M:%S"),file=txt)

#Read data
x_num = 4200#Number of wavelengths
soil_texture = 4206#The number of columns where the soil_texture is located
df=pd.read_csv(r'LUCAS_Soil_Dataset.csv',encoding='GB2312')#Read data that has been randomly shuffled
data = df.iloc[:,1:soil_texture].values

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
train_soil_texture = train[:,soil_texture-2]
train_one_hot = pd.get_dummies(train_soil_texture)
test_soil_texture = test[:,soil_texture-2]
test_one_hot = pd.get_dummies(test_soil_texture)
print(test_one_hot[:10],file=txt)

input_shape = (1, x_num, 1)#input_shape = (1, 4200, 1)

#Build model 
model = Sequential()   
model.add(Conv2D(6,(1,3),strides=(1,1),input_shape=input_shape,padding='same',activation='tanh'))
model.add(Conv2D(6,(1,3),strides=(1,1),padding='same',activation='tanh'))
model.add(MaxPooling2D(pool_size=(1,2),strides=(1,2)))
    
model.add(Conv2D(12,(1,3),strides=(1,1),padding='same',activation='tanh'))
model.add(Conv2D(12,(1,3),strides=(1,1),padding='same',activation='tanh'))
model.add(MaxPooling2D(pool_size=(1,2),strides=(1,2)))

model.add(Conv2D(24,(1,3),strides=(1,1),padding='same',activation='tanh'))
model.add(Conv2D(24,(1,3),strides=(1,1),padding='same',activation='tanh'))
model.add(Conv2D(24,(1,3),strides=(1,1),padding='same',activation='tanh'))
model.add(MaxPooling2D(pool_size=(1,2),strides=(1,2)))

model.add(Conv2D(48,(1,3),strides=(1,1),padding='same',activation='tanh'))
model.add(Conv2D(48,(1,3),strides=(1,1),padding='same',activation='tanh'))
model.add(Conv2D(48,(1,3),strides=(1,1),padding='same',activation='tanh'))
model.add(MaxPooling2D(pool_size=(1,2),strides=(1,2)))
    
model.add(Conv2D(48,(1,3),strides=(1,1),padding='same',activation='tanh'))
model.add(Conv2D(48,(1,3),strides=(1,1),padding='same',activation='tanh'))
model.add(Conv2D(48,(1,3),strides=(1,1),padding='same',activation='tanh'))
model.add(MaxPooling2D(pool_size=(1,2),strides=(1,2)))
   
model.add(Flatten())
model.add(Dense(200,activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(100,activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(4,activation='sigmoid'))#Four groups of classification
#model.add(Dense(12,activation='sigmoid'))#12 levels of classification
model.summary()

#Setting parameters, compile and fit
nadam = optimizers.Nadam(lr=0.0001, epsilon=1e-08)
early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=2)
model.compile(optimizer=nadam,loss='categorical_crossentropy',metrics=['accuracy'])
hist = model.fit(train_x,train_one_hot,epochs=10000,batch_size=32,validation_split=0.2,verbose=2, shuffle=True, callbacks=[early_stopping])

#predict and save
predict = model.predict(test_x)
predicted_class=predict.argmax(axis=1)
true_class = np.array(test_one_hot).argmax(axis=1)
print("prediction accuracy: {}".format(sum(predicted_class==true_class)/len(true_class)),file=txt)
pre=pd.DataFrame(data=predicted_class)
pre.to_csv('LucasVGGNet_16_predict_values.csv')

#Graph of loss change during training
plt.figure()
plt.plot(np.arange(0, len(hist.history["loss"])), hist.history["accuracy"], label="Calibration")
plt.plot(np.arange(0, len(hist.history["loss"])), hist.history["val_accuracy"], label="Validation")
plt.title("Calibration/Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.savefig("LucasVGGNet_16_accuracy.jpg",dpi=300)

#Save the model
model.save('LucasVGGNet_16.h5')

#Running time
end = time.time()
print (time.strftime(f"%Y-%m-%d %H:%M:%S"),file=txt)
print("Running time:%.2f秒"%(end-start),file=txt)
txt.close()
