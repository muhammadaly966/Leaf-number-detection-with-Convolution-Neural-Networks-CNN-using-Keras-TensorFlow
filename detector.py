#!/usr/bin/env python
# coding: utf-8

#   # Name:Saad,Mohammad
# 
#   # Student no :300267006                                                  
#  
#  #  Assignment (3)
# 
#   # ELG7186[EI] Learning-Based Computer Vision 

# In[45]:


import numpy as np
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import skimage.io as skio
from skimage.util import crop
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay,mean_squared_error
import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Dropout,BatchNormalization
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers.experimental import preprocessing
from keras.preprocessing.image import ImageDataGenerator
tensorflow.random.set_seed(42)
np.random.seed(42)


# ## Data Preparation
# 

# In the next cell some functions were defined to read, resize and plot the images.

# In[3]:


#images directory
image_directory_training="./training"
image_directory_testing="./testing"
#func to read each image and resize it to be 128*128
def img_process(path):
    img_read=skio.imread(path) # to read each img
    img_resized= resize(img_read,(128,128),preserve_range=True,anti_aliasing=True,order=0)
    return img_resized 
#func to ploy gray scale images
def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")
#func to ploy gray colored images
def plot_color_image(image):
    plt.imshow(image.astype(np.uint8),interpolation="nearest")
    plt.axis("off")
#func to load all the dataset 
def load_label_resize(image_directory):
    files=os.listdir(image_directory)
    x_1=[]
    x_2=[]
    x_3=[]
    x_4=[]
    x_6=[]
    for i in files : #to open each file and read images
        path=os.path.join(image_directory,i)
        if i == "1":
            for img in os.listdir(path):
                x_1.append(img_process(os.path.join(path,img)))
        elif i == "2":
            for img in os.listdir(path):
                x_2.append(img_process(os.path.join(path,img)))
        
        elif i == "3":
            for img in os.listdir(path): 
                x_3.append(img_process(os.path.join(path,img)))
        
        elif i == "4":
            for img in os.listdir(path):
                x_4.append(img_process(os.path.join(path,img)))
            
            
        elif i == "6":
            for img in os.listdir(path):
                x_6.append(img_process(os.path.join(path,img)))
        
        X_1=np.array(x_1)
        X_1=X_1.reshape(X_1.shape[0],128,128,3)
        X_2=np.array(x_2)
        X_2=X_2.reshape(X_2.shape[0],128,128,3)
        X_3=np.array(x_3)
        X_3=X_3.reshape(X_3.shape[0],128,128,3)
        X_4=np.array(x_4)
        X_4=X_4.reshape(X_4.shape[0],128,128,3)
        X_6=np.array(x_6)
        X_6=X_6.reshape(X_6.shape[0],128,128,3)
                
    return X_1,X_2,X_3,X_4,X_6
  


# Loading training and testing dataset

# In[4]:


x_1,x_2,x_3,x_4,x_6=load_label_resize(image_directory_training)#training
x_1_t,x_2_t,x_3_t,x_4_t,x_6_t=load_label_resize(image_directory_testing)#testing


# Creating the images labels for the **classification network**<br>
# the classes were labeled as following:<br>
#     1-leaf >> class 0 <br>
#     2-leaves >> class 1 <br>
#     3-leaves >> class 2 <br>
#     4-leaves >> class 3 <br>
#     6-leaves >> class 4 <br>

# In[5]:


#labeling the training data for classification
y_1=np.zeros_like(x_1[:,0,0,0])  #the 1 leaves category is labeled class 0
y_2=np.ones_like(x_2[:,0,0,0])   #the 2 leaves category is labeled class 1
y_3=np.ones_like(x_3[:,0,0,0])*2 #the 3 leaves category is labeled class 2
y_4=np.ones_like(x_4[:,0,0,0])*3 #the 4 leaves category is labeled class 3
y_6=np.ones_like(x_6[:,0,0,0])*4 #the 6 leaves category is labeled class 4

#labeling the testing data for classification
y_1_t=np.zeros_like(x_1_t[:,0,0,0])
y_2_t=np.ones_like(x_2_t[:,0,0,0])
y_3_t=np.ones_like(x_3_t[:,0,0,0])*2
y_4_t=np.ones_like(x_4_t[:,0,0,0])*3
y_6_t=np.ones_like(x_6_t[:,0,0,0])*4 


# Creating the images labels for the **regression network**<br>
# here each class was labeled with it's actual value because it's numerical regression and numbers has actual meaning for the process.
# 

# In[6]:


#labeling the training data for regression
y_1_reg=np.ones_like(x_1[:,0,0,0])  
y_2_reg=np.ones_like(x_2[:,0,0,0])*2   
y_3_reg=np.ones_like(x_3[:,0,0,0])*3 
y_4_reg=np.ones_like(x_4[:,0,0,0])*4 
y_6_reg=np.ones_like(x_6[:,0,0,0])*6 

#labeling the testing data for regression
y_1_t_reg=np.ones_like(x_1_t[:,0,0,0])
y_2_t_reg=np.ones_like(x_2_t[:,0,0,0])*2
y_3_t_reg=np.ones_like(x_3_t[:,0,0,0])*3
y_4_t_reg=np.ones_like(x_4_t[:,0,0,0])*4
y_6_t_reg=np.ones_like(x_6_t[:,0,0,0])*6 


# In[7]:


#stacking the classes together for classification
X=np.vstack((x_1,x_2,x_3,x_4,x_6))
y_class=np.hstack((y_1,y_2,y_3,y_4,y_6))
X_test=np.vstack((x_1_t,x_2_t,x_3_t,x_4_t,x_6_t))
y_test_class=np.hstack((y_1_t,y_2_t,y_3_t,y_4_t,y_6_t))


# In[8]:


#stacking the classes together for regression
y_reg=np.hstack((y_1_reg,y_2_reg,y_3_reg,y_4_reg,y_6_reg))
y_test_reg=np.hstack((y_1_t_reg,y_2_t_reg,y_3_t_reg,y_4_t_reg,y_6_t_reg))


# The training dataset was splitted for **training** and **validation**  to train the **classification network**

# In[9]:


X_train, X_val, y_train_class, y_val_class = train_test_split(X, y_class, test_size=0.2, stratify=y_class, random_state=42)


# In[44]:


print('number of exampels used in trainig: ', X_train.shape[0])
print('number of exampels used in validation: ', X_val.shape[0])


# The VGG16 model was loaded without the top block

# In[10]:


vgg_model = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(128, 128, 3))
print(vgg_model.summary())

vgg_model.trainable = False


# Our model was built using the first two blocks from VGG16 and 2 fully connected layers with each has 256 units and an output layers with 5 units ,one for each class prediction

# In[11]:


x = vgg_model.layers[-13].output
# Flatten as before
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(5, activation='softmax')(x)
vgg_model_transfer = Model(inputs=vgg_model.input, outputs=x)


# In[12]:


print(vgg_model_transfer.summary())


# As the previously printed summary shows that the structure of the model matches what was descriped above.

# In[11]:


nClasses = 5
# Use Keras' handy utils
y_train_k = tensorflow.keras.utils.to_categorical(y_train_class, num_classes=nClasses)
y_val_k = tensorflow.keras.utils.to_categorical(y_val_class, num_classes=nClasses)


# In the above cell the labels was binary encoded to be suitable for training the model <br>
# the next cell tests the encoding process and prints the results

# In[46]:


for i in range(0,250,20):
    print(y_train_class[i], " ", y_train_k[i,:])


# Our model was trained with batch size = 128 and epochs number = 30 

# In[15]:


batchSize = 128
nEpochs = 35

    
optimizer = keras.optimizers.Adam(epsilon=0.0001)
vgg_model_transfer.compile(loss='categorical_crossentropy', 
                           optimizer=optimizer, 
                           metrics=['accuracy'])


history_class = vgg_model_transfer.fit(X_train, y_train_k, batch_size=batchSize, epochs=nEpochs, verbose=1, 
                                 validation_data=(X_val, y_val_k))


# As the results shows the model reaches accuracy around 99% and validation accuracy around 40% <br>
# The learning curves were plotted below

# In[16]:


history_frame_class = pd.DataFrame(history_class.history)
history_frame_class.loc[5:, ['loss', 'val_loss']].plot()
history_frame_class.loc[:, ['accuracy', 'val_accuracy']].plot();


# The model classification report and confusion matrix were plotted for **training dataset**

# In[17]:


y_predict_prob = vgg_model_transfer.predict(X)
y_predict = y_predict_prob.argmax(axis=-1)
#trainign dataset
print(classification_report(y_class, y_predict, target_names=['1','2','3','4','6']))
cm=confusion_matrix(y_class,y_predict)
ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['1','2','3','4','6']).plot()


# The model classification report and confusion matrix were plotted for **testing dataset**

# In[18]:


y_predict_prob_test = vgg_model_transfer.predict(X_test)
y_predict_test = y_predict_prob_test.argmax(axis=-1)
#testing dataset
print(classification_report(y_test_class, y_predict_test, target_names=['1','2','3','4','6']))
cm=confusion_matrix(y_test_class,y_predict_test)
ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['1','2','3','4','6']).plot()


# The data was splitted again but with using the regression labels

# In[86]:


X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(X, y_reg, test_size=0.2, stratify=y_reg, random_state=42)


# A model similar to the one used in classification was built with one output layer

# In[87]:


r = vgg_model.layers[-13].output
r = Flatten()(r)
r = Dense(256, activation='relu')(r)
r = Dense(512, activation='relu')(r)
r = Dense(1024, activation='relu')(r)
r = Dense(2048, activation='relu')(r)
r = Dense(1)(r)
vgg_model_transfer_regression = Model(inputs=vgg_model.input, outputs=r)
print(vgg_model_transfer_regression.summary())


# Training the model in a way similar to the classifiation with using loss as mean square error 

# In[88]:


batchSize = 128
nEpochs = 35

    
optimizer = keras.optimizers.Adam(epsilon=0.0001)
vgg_model_transfer_regression.compile(loss='mse', 
                           optimizer=optimizer)


history_reg = vgg_model_transfer_regression.fit(X_train_reg, y_train_reg, batch_size=batchSize, epochs=nEpochs, verbose=1, 
                                 validation_data=(X_val_reg, y_val_reg))


# In[89]:


history_frame_reg = pd.DataFrame(history_reg.history)
history_frame_reg.loc[15:, ['loss', 'val_loss']].plot()


# In[91]:


y_pred = vgg_model_transfer_regression.predict(X)
print("the mean square error on the total training data is {} ".format(mean_squared_error(y_reg, y_pred)))
y_pred_test = vgg_model_transfer_regression.predict(X_test)
print("the mean square error on the testing data is {} ".format(mean_squared_error(y_test_reg, y_pred_test)))


# ## Discussion

# -By analyzing the results from the learning curves at **Question 3.1**, it appears that the model is overfitting the training data with loss approximately zero and accuracy over 99% but looking at the validation loss at first it decreases but after around 16 epochs it tends to be stable at a high number without any improvement and the same happens to accuracy which increases and then reaches a constant value around 40% <br>
# All of that indicates to model overfitting on the splitted training data<br>
# -The overall performance of the model on the total trainig dataset was checked using classification report and confusion matrix<br>
# -The model on the training data got high accuracy equals to 88% and it was performing best in detecting plant with one leaf <br>
# -The performance on testing data was low as expected due to overfitting.The model accuracy was 40% and still detecting the first class better than the others.<br>
# <br>
# <br>
# -At **Question 3.2** the results from the learning curves suggest low performance and the mean square error of the model on training data is high and equals 1688.4 and on the testing data equals 853.6<br>
# A lot of factors can be probably the reason for that to happen :<br>
# * the model can be underfitting the data and needs to increase the capacity of the model to solve such a problem<br>
# * the dataset size is small and increasing the dataset can cause some improvemnt in the performance<br>
# 
# *side note:For **(Q 3.2 )** I tried several compinations for the layers and it didn't help the model to converge any better*<br>
# **from the brevious results it appears that the model in Q 3.1 performs better and can be expected to improved using regularization and data augmentation**
# 
# 
# 
#   

# ## Improving the Model

# ### Regularization
# to regularize the weights and prevent the model from overfitting 3 layers of batch normalization and dropout were added to the classification model.

# In[75]:


x = vgg_model.layers[-13].output
x = Flatten()(x)
x = BatchNormalization()(x)
x = Dropout(rate=0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(rate=0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(rate=0.5)(x)
x = Dense(5, activation='softmax')(x)
vgg_model_transfer_regularized = Model(inputs=vgg_model.input, outputs=x)
print(vgg_model_transfer_regularized.summary())


# The model was trained using the same previous conditions :

# In[76]:


batchSize = 128
nEpochs = 35

    
optimizer = keras.optimizers.Adam(epsilon=0.0001)
vgg_model_transfer_regularized.compile(loss='categorical_crossentropy', 
                           optimizer=optimizer, 
                           metrics=['accuracy'])


history_class_reg = vgg_model_transfer_regularized.fit(X_train, y_train_k, batch_size=batchSize, epochs=nEpochs, verbose=1, 
                                 validation_data=(X_val, y_val_k))


# In the next cell the learning curves of the model was plotted  and it can be noticed that the gap between the training losse and validation loss is more less than the previous model

# In[77]:


history_frame_class_reg = pd.DataFrame(history_class_reg.history)
history_frame_class_reg.loc[5:, ['loss', 'val_loss']].plot()
history_frame_class_reg.loc[:, ['accuracy', 'val_accuracy']].plot();


# The classification report and confusion matrix was printed for the trainig data and for the testing data :

# In[78]:


y_predict_prob = vgg_model_transfer_regularized.predict(X)
y_predict = y_predict_prob.argmax(axis=-1)
#trainign dataset
print(classification_report(y_class, y_predict, target_names=['1','2','3','4','6']))
cm=confusion_matrix(y_class,y_predict)
ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['1','2','3','4','6']).plot()


# In[79]:


y_predict_prob_test = vgg_model_transfer_regularized.predict(X_test)
y_predict_test = y_predict_prob_test.argmax(axis=-1)
#testing dataset
print(classification_report(y_test_class, y_predict_test, target_names=['1','2','3','4','6']))
cm=confusion_matrix(y_test_class,y_predict_test)
ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['1','2','3','4','6']).plot()


# A slight improvement can be noticed in the above displayed data.

# ### Data Augmentation

# Due to the small amount of dataset the model always tends to overfit the trainging data and a solution to that is to augment the data by generating some modification on the original images in order to teach the model better<br>
# 
# Each time an image enters the model it will be subjected to several random processes from ( flipping , translation ,rotation and contrast) and it will have a random compination of all these processes.<br>

# In[80]:


model_augmented = keras.Sequential([layers.InputLayer(input_shape=[128, 128, 3]),
layers.RandomFlip(mode="horizontal_and_vertical"),
layers.RandomTranslation(
    height_factor=0.1,
    width_factor=0.1,
    fill_mode="reflect",
    interpolation="bilinear"),   
layers.RandomRotation(
    factor=0.3,
    fill_mode="reflect",
    interpolation="bilinear",
    seed=None,
    fill_value=0.0,),                                    
layers.RandomContrast(factor=0.3),                                    
vgg_model_transfer_regularized])
print(model_augmented.summary())


# The model was trained as the previous models :
# 

# In[82]:


batchSize = 128
nEpochs = 35

    
optimizer = keras.optimizers.Adam(epsilon=0.0001)
model_augmented.compile(loss='categorical_crossentropy', 
                           optimizer=optimizer, 
                           metrics=['accuracy'])


history_class_aug = model_augmented.fit(X_train, y_train_k, batch_size=batchSize, epochs=nEpochs, verbose=1, 
                                 validation_data=(X_val, y_val_k))


# In the next cell the learning curves of the model was plotted  and it can be noticed that the gap between the training losse and validation loss is so small and it fluctuates around 0.1 

# In[85]:


history_frame_class_aug = pd.DataFrame(history_class_aug.history)
history_frame_class_aug.loc[:, ['loss', 'val_loss']].plot()
history_frame_class_aug.loc[:, ['accuracy', 'val_accuracy']].plot();


# The classification report and confusion matrix was printed for the trainig data and for the testing data :

# In[84]:


y_predict_prob = model_augmented.predict(X)
y_predict = y_predict_prob.argmax(axis=-1)
#trainign dataset
print(classification_report(y_class, y_predict, target_names=['1','2','3','4','6']))
cm=confusion_matrix(y_class,y_predict)
ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['1','2','3','4','6']).plot()


# In[83]:


y_predict_prob_test = model_augmented.predict(X_test)
y_predict_test = y_predict_prob_test.argmax(axis=-1)
#testing dataset
print(classification_report(y_test_class, y_predict_test, target_names=['1','2','3','4','6']))
cm=confusion_matrix(y_test_class,y_predict_test)
ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['1','2','3','4','6']).plot()


# ## Discussion

# -By analyzing the results from the learning curves at **Question 4.1**, it appears that the model is also overfitting the training data with loss approximately 0.015 and accuracy over 99%  which is similar to the previous classificaton model(Q3.1) but looking at the validation loss the values now are less than the case without regularization and the gap between trainig loss and validation loss are much smaller.These results suggest slight improvement in the model performance on the testing data.<br>
# <br>
# -The overall performance of the model on the total trainig dataset was checked using classification report and confusion matrix<br>
# -The model on the training data got higher accuracy equals to 90% and it was performing best in detecting plant with one leaf <br>
# -The performance on testing data was improved also as expected due to regularization.The model accuracy increased to 45% and still detecting the first class better than the others.<br>
# <br>
# <br>
# -At **Question 5** the results from the learning curves showed no evidence of overfitting.The training loss was always close the validation loss which suggested that the model is now more generalized as if we have used larger datasets. <br>
# <br>
# -The overall performance of the model on the total trainig dataset was checked using classification report and confusion matrix<br>
# -The model on the training data got accuracy equals 86% which is less than the regularized model and the original model  ,this was expected to happen on the training data because the model now is less overfittting.<br>
# 
# -The performance on testing data was improved also as expected due to the data augmentation.The model accuracy improved to be 56% and still detecting the first class better than the others but now it detects the 6 leaves class with higher performance than before.<br>
# 
# **from the brevious results one can deduce that the problem of having small dataset can be solved using :<br>
# * Regularization to prevent overfitting the small number of data 
# * Augmentation to teach the model with more discriptive dataset and to have more generalized model <br>
# 
# *side note: form our experience here the effect of data augmentation was larger and more dominant in terms of metrics values.*
# 
# 
# 
# 
# 
#   
