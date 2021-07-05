import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import shap
import time

#Read data
start = time.time()
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
df=pd.read_csv(r'LUCAS_Soil_Dataset.csv')#Read data that has been randomly shuffled
x_num = 4200#Number of wavelengths
data = df.iloc[:,1:x_num+1].values

#Divide training set and testing set
ratio = int(0.75*len(data))
train = data[:ratio]
test = data[ratio:]
train_x = train[:,:x_num]
train_x = train_x.reshape(train_x.shape[0], 1, x_num, 1)    
test_x = test[:,:x_num]
test_x = test_x.reshape(test_x.shape[0], 1, x_num, 1)

#load_model
model = load_model('LucasResNet_16.h5')

#Calculate SHAP values
background = train_x[np.random.choice(train_x.shape[0], 500, replace=False)]
e = shap.DeepExplainer(model,background)
shap_values = e.shap_values(test_x)#or train_x 
test_x=test_x[:,0,:,0]
shap_values=np.array(shap_values)
shap_values=shap_values[0,:,0,:,0]

#Export "feature_sort_dot.jpg"
feature_list = []
for i in np.arange(400,2500,0.5):
    feature_list.append(i)
    
shap.summary_plot(shap_values,test_x,plot_type="dot",max_display=100,show=False,feature_names=feature_list)
plt.savefig("feature_sort_dot.jpg",dpi=300)
plt.show()

#Export "Wavelengths_SHAP_mean_absolute_value.csv"
Wavelengths_shap=abs(shap_values)
Wavelengths_shap = np.mean(Wavelengths_shap, axis=0)
Wavelengths_shap_value=pd.DataFrame(data=Wavelengths_shap)
Wavelengths_shap_value.to_csv('Wavelengths_SHAP_mean_absolute_value.csv')

#Export "SHAP_mean_absolute_value.jpg"
plt.figure()
plt.plot(Wavelengths_shap_value,color='black',linewidth=0.5)
plt.xlabel("Wavelength(nm)")
plt.ylabel("SHAP mean absolute value")
plt.savefig("SHAP_mean_absolute_value.jpg",dpi=300)

#Running time
end = time.time()
print (time.strftime(f"%Y-%m-%d %H:%M:%S,{time.localtime()}"))
print("Running time:%.2f秒"%(end-start))