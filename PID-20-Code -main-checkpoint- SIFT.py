#!/usr/bin/env python
# coding: utf-8

# # Traffic Detection on Road - PID20

# In[2]:


import numpy as np                                    # to perform a wide variety of mathematical operations on arrays.                                             
import pandas as pd                                   # pandas is used to analyze data.

import os                                             # a portable way of using operating system dependent functionality.
import csv                                            # implements classes to read and write tabular data in CSV format.
import cv2                                            # for image processing and performing computer vision tasks.
import pickle                                         # to dump and load data
import seaborn as sns                                 # for making statistical graphics

from skimage.io import imread                         # to read an image 
from skimage.transform import resize                  # It is used to Warp an image according to a given coordinate transformation. 
import matplotlib.pyplot as plt                       # plotting

from sklearn.model_selection import train_test_split  # to split data in training and testing
from sklearn.metrics import classification_report     # to get classification report
import sklearn.metrics as metrics                     # get all metrices functions
from sklearn.metrics import accuracy_score            # to compute accuracy


from sklearn.metrics import confusion_matrix          # to get confusion matrix for testing data
from datetime import datetime                         # to calculate time
from sklearn import linear_model               

from sklearn.cluster import KMeans                    # to import Kmeans Clusterring
from sklearn.model_selection import GridSearchCV      # to import Grid method for training
from sklearn.neighbors import KNeighborsClassifier    # to import KNN
from sklearn.svm import LinearSVC                     # to import SVC with linear kernel
from sklearn.svm import SVC                           # to import SVC
from sklearn.tree import DecisionTreeClassifier       # to import Decision tree classifier
from sklearn.ensemble import RandomForestClassifier   # to import random forest classifier
from sklearn.neighbors import KNeighborsClassifier    
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import KFold    
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Normalizer          # to normalize sift feature extracted

from sklearn.naive_bayes import GaussianNB
import joblib;


# # Preprocessing and feature description

# In[2]:


input0=r"F:\Academic folders\TYSEM1\EDI\EDI_DATASET\Staircaseold"
input1=r"F:\Academic folders\TYSEM1\EDI\EDI_DATASET\Potholeold"


# In[8]:


# For Class of type 1 with index i = 0

i=0
for filename in os.listdir(input0):
   
    path=os.path.join(input0,filename)
    in0=cv2.imread(path)
    path
    print("Input Image : ", input0, i)

    #plt.imshow(in0, cmap = 'gray')
    #plt.xlabel('X axis')
    #plt.ylabel('Y axis')
    #plt.title('Input Gray Image')
    #plt.show()
    
    #resize image
    resize=(512,512)
    img0=cv2.resize(in0,resize)
    gray0=cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

        
    #plt.imshow(gray0, cmap = 'gray')
    #plt.xlabel('X axis')
    #plt.ylabel('Y axis')
    #plt.show()
    #gray image
    path= r"F:\\Academic folders\\TYSEM1\\EDI\\EDI_DATASET\\Prepocold\\Staircase\\"
    cv2.imwrite(os.path.join(path, 'Staircase'+ str(i) + '.jpg'), gray0)
    i=i+1

cv2.waitKey(0)


# In[9]:


# For Class of type 1 with index i = 0

i=0
for filename in os.listdir(input1):
   
    path=os.path.join(input1,filename)
    in0=cv2.imread(path)
    path
    print("Input Image : ", input1, i)

    #plt.imshow(in0, cmap = 'gray')
    #plt.xlabel('X axis')
    #plt.ylabel('Y axis')
    #plt.title('Input Gray Image')
    #plt.show()
    
    #resize image
    resize=(512,512)
    img0=cv2.resize(in0,resize)
    gray0=cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

        
    #plt.imshow(gray0, cmap = 'gray')
    #plt.xlabel('X axis')
    #plt.ylabel('Y axis')
    #plt.title('Resized Image:')
    #plt.show()
    
    #gray image
    path= r"E:\\TYSEM1\\EDI\\EDI_DATASET\\Prepoc\\Pothole\\"
    cv2.imwrite(os.path.join(path, 'Pothole'+ str(i) + '.jpg'), gray0)
    i=i+1

cv2.waitKey(0)


# In[3]:


import cv2
import pandas as pd
import numpy as np
import os
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


folder1=r"F:\Academic folders\TYSEM1\EDI\EDI_DATASET\Prepoc\Staircase"
folder2=r"F:\Academic folders\TYSEM1\EDI\EDI_DATASET\Prepoc\Pothole"


# ## SIFT Descriptors
# ### Staircase

# In[11]:


i=0
for filename in os.listdir(folder1):
    #path
    path=os.path.join(folder1,filename)
    a=cv2.imread(path)
    
    #resize image
    resize=(512,512)
    img=cv2.resize(a,resize)
    
    #gray image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #initialise orb descriptor
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    #convert the descriptor array into a dataframe format
    out=pd.DataFrame(descriptors)
    print("descriptor shape ",i," : ", out.shape)
    i=i+1
    
    #drop first coloumn as it's the no of feature detected. Not required.
    #append to the csv file
    csv_data=out.to_csv(r'F:\Academic folders\TYSEM1\EDI\EDI_DATASET\SIFT_1000\Staircase.csv', mode='a', header=False,index=False)
    
    


# In[9]:


data1 = pd.read_csv(r'F:\Academic folders\TYSEM1\EDI\EDI_DATASET\SIFT_1000\Staircase.csv',header=None,dtype='uint8')

data1=data1.astype(np.uint8) 
#as unit8 contain 1 byte(8bit) and ranges from 0 to 255
data1


# ## Pothole
# 

# In[12]:


i=0
for filename in os.listdir(folder2):
    #path
    path=os.path.join(folder2,filename)
    a=cv2.imread(path)
    
    #resize image
    resize=(512,512)
    img=cv2.resize(a,resize)
    
    #gray image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #initialise orb descriptor
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    #convert the descriptor array into a dataframe format
    out=pd.DataFrame(descriptors)
    print("descriptor shape ",i," : ", out.shape)
    i=i+1
    
    #drop first coloumn as it's the no of feature detected. Not required.
    #append to the csv file
    csv_data=out.to_csv(r'F:\Academic folders\TYSEM1\EDI\EDI_DATASET\SIFT_1000\Pothole.csv', mode='a', header=False,index=False)


# In[10]:



data2= pd.read_csv(r'F:\Academic folders\TYSEM1\EDI\EDI_DATASET\SIFT_1000\Pothole.csv',header=None,dtype='uint8')
data2=data2.astype(np.uint8)
data2


# In[12]:


data=data1.append(data2)


# In[5]:


csv_data=data.to_csv(r'F:\Academic folders\TYSEM1\EDI\EDI_DATASET\SIFT_1000\Sift_Final.csv', mode='a', header=False,index=False)


# In[13]:


data


# # Elbow method

# In[19]:


# Find K value in Kmeans :- we will used elbow method
SSE = []                                      # list to store inertia for each value of K

for k in range(1,20):                         # to iterate through k value from 1 to 14
    print(k)                              
    KM_Model = KMeans(n_clusters = k )        # initialize Kmeans clustering model for cluster count of K
    KM_Model.fit(data)                        # fit ORB feature data to KMEans clustering model
    SSE.append(KM_Model.inertia_)             # append inertia to list
    
SSE                                           # print value of inertia


# #### Elbow plot

# In[20]:


# Plot between Acc and K value.
#X_Axis = K value
#Y_Axis = SSE
import matplotlib.pyplot as plt
plt.plot(range(1,20),SSE,'*-')
plt.xlabel("K_Value")
plt.ylabel("Acc")
# As the steepness between the k value of 7 and 9 changes significantly k value or number of cluster is chosen to be 8.


# # Kmeans

# In[18]:


kmeans = KMeans(n_clusters=6)
kmeans.fit(data)


# In[19]:


import pickle


# In[16]:


# save the model to disk
filename = 'Kmeans_CL_5_Model.sav'
pickle.dump(kmeans, open(filename, 'wb'))


# In[20]:


hist=np.histogram(kmeans.labels_,bins=[0,1,2,3,4,5,6])


print('histogram of trained kmeans')
print(hist,"\n")


# In[16]:


#folder1=r"D:/StrData/Str"
#folder2=r"D:\StrData\Nstr"


# # Predicting and normalizing predictions

# ###  predictions

# In[27]:


#performing kmeans prediction of the entire Staircase dataset with the pretrained kmeans model

#initialising i=0; as its the first class
i=1
j=0
data=[]
k=0

for filename in os.listdir(folder1):
    #path
    path=os.path.join(folder1,filename)
    a=cv2.imread(path)
    
    #resize image
    resize=(512,512)
    img=cv2.resize(a,resize)
    
    #gray image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    out=pd.DataFrame(descriptors)
    
    #predict values of feature vector with pretrained kmeans
    #ValueError: Buffer dtype mismatch, expected 'float', in order to avoid this dtype=np.double
    
    array_double = np.array(out, dtype=np.double)
    a=kmeans.predict(array_double)
    
    j=j+1
    hist=np.histogram(a,bins=[0,1,2,3,4,5,6])
    print("feature vector of  ",j," : ", hist[0])
    
    #append the dataframe into the array in append mode, the array will only have 5 values which will store the values in a row
    data.append(hist[0])
    k=k+1
    
norm = Normalizer()
normalized = norm.fit_transform(data)                   # normalize the histogram 

#convert Array to Dataframe and append to the list
Output = pd.DataFrame(data)
#add row class 
Output["Class"] = i 
csv_data=Output.to_csv(r'F:\Academic folders\TYSEM1\EDI\EDI_DATASET\kmeansstaircase.csv', mode='a',header=False,index=False)


# ###  Predictions

# In[28]:


#performing kmeans prediction of the entire Staircase dataset with the pretrained kmeans model

#initialising i=0; as its the first class
i=2
j=0
data=[]
k=0

for filename in os.listdir(folder2):
    #path
    path=os.path.join(folder2,filename)
    a=cv2.imread(path)
    
    #resize image
    resize=(512,512)
    img=cv2.resize(a,resize)
    
    #gray image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    out=pd.DataFrame(descriptors)
    
    #predict values of feature vector with pretrained kmeans
    #ValueError: Buffer dtype mismatch, expected 'float', in order to avoid this dtype=np.double
    
    array_double = np.array(out, dtype=np.double)
    a=kmeans.predict(array_double)
    
    j=j+1
    hist=np.histogram(a,bins=[0,1,2,3,4,5,6])
    print("feature vector of  ",j," : ", hist[0])
    
    #append the dataframe into the array in append mode, the array will only have 5 values which will store the values in a row
    data.append(hist[0])
    k=k+1
    
norm = Normalizer()
normalized = norm.fit_transform(data)                   # normalize the histogram 

#convert Array to Dataframe and append to the list
Output = pd.DataFrame(data)
#add row class 
Output["Class"] = i 
csv_data=Output.to_csv(r'F:\Academic folders\TYSEM1\EDI\EDI_DATASET\kmeanspothole.csv', mode='a',header=False,index=False)


# # Displaying the kmeans predicted data

# In[21]:


#Displaying the kmeans predicted data
print("Staircase")
dat1= pd.read_csv(r'F:\Academic folders\TYSEM1\EDI\EDI_DATASET\kmeansstaircase.csv',header=None)
print(dat1)


# In[22]:


print("Pothole")
dat2= pd.read_csv(r'F:\Academic folders\TYSEM1\EDI\EDI_DATASET\kmeanspothole.csv',header=None)
print(dat2)


# In[23]:


A=dat1.append(dat2)


# In[31]:


#appending All classes into 1 csv file
csv_data=A.to_csv(r'F:\Academic folders\TYSEM1\EDI\EDI_DATASET\FinalFV.csv', mode='a',header=False,index=False)


# In[24]:


x = A.iloc[:,0:6].values
x
y = A.iloc[:,6].values
y


# In[25]:


from sklearn.preprocessing import StandardScaler
DhoniS = StandardScaler()
x_scaled = DhoniS.fit_transform(x)


# In[26]:


x_scaled


# # Dimensionality reduction using PCA

# In[27]:


from sklearn.decomposition import PCA


# In[28]:


pca = PCA(n_components=None)
pca.fit(x_scaled)


# In[29]:


x_pca = pca.transform(x_scaled)


# In[30]:


x_pca


# In[31]:


x_pca = pd.DataFrame(x_pca)


# In[32]:


x_pca


# In[33]:


x_pca.shape


# ## Calculating number of components required

# In[34]:


print(pca.explained_variance_ratio_) 


# In[35]:


print(pca.explained_variance_ratio_.sum()) 


# In[36]:


l = pca.explained_variance_ratio_
l = l[:4]
l.sum()   


# In[37]:


pca = PCA(n_components=4) #91.54 % variation explaned
pca.fit(x_scaled)


# In[38]:


# save the model to disk
filename = 'PCA_3_Model.sav'
pickle.dump(pca, open(filename, 'wb'))


# ### Reducing the dimensions 

# In[39]:


x_pca = pca.transform(x_scaled)


# In[40]:


x_pca


# In[41]:


x_pca.shape


# In[42]:


print(pca.explained_variance_ratio_) 


# In[43]:


x_pca = pd.DataFrame(x_pca)


# In[44]:


x_pca


# In[45]:


B=pd.concat([x_pca, pd.DataFrame(y)],axis=1)


# In[46]:


B


# In[47]:


csv_data=B.to_csv(r'F:\Academic folders\TYSEM1\EDI\EDI_DATASET\FinalPCAFV.csv', mode='a',header=False,index=False)


# In[ ]:





# # Training ,Testing & Classification

# In[48]:


import pandas as pd  
data= pd.read_csv(r'F:\Academic folders\TYSEM1\EDI\EDI_DATASET\FinalPCAFV.csv',header=None)

data


# In[49]:


#assigning x the columns from 1 to 128 for training
x = data.iloc[:,0:4].values
print("X values")
print(x)

#assigning y with the column "Class" as target variable
y = data.iloc[:,4]
print("Y values")
print(y)


# In[50]:


#Dataset split into train and test with 80% Training and 20% Testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=0)


# # Classification using 3 classifiers

# ### 1)Decision Tree Classifier

# In[51]:


#Decision Tree Classifier
model1 = DecisionTreeClassifier(max_depth=9)
model1.fit(x_train, y_train)
y_pred1 = model1.predict(x_test)
CM = confusion_matrix(y_test, y_pred1)
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
TNR = TN/(TN+FP) # Specificity or true negative rate
FPR = FP/(FP+TN) # Fall out
print("Decision Tree Results")
print("Train Accuracy:",model1.score(x_train, y_train))
print("Test Accuracy:",model1.score(x_test, y_test))
print("Precision Score: ",metrics.precision_score(y_test, y_pred1))
print("Recall Score: ",metrics.recall_score(y_test, y_pred1)) # true positive rate, Sensitivity
print("True Negative Rate: ", TNR)
print("False Positive Rate: ", FPR)
print("F2 Score: ",metrics.fbeta_score(y_test, y_pred1, beta=2.0))
print("F1 Score: ",metrics.f1_score(y_test, y_pred1))
print("Sensitivity: ",TP/(TP+FN))
print("Specificity: ",TNR)
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred1))
print("ROC curve ",metrics.plot_roc_curve(model1,x_test, y_test))
filename = 'DT_Model.pkl'
pickle.dump(model1, open(filename, 'wb'))


# ### 2)Random Forest Clasifier

# In[52]:


from sklearn.ensemble import RandomForestClassifier

model2 = RandomForestClassifier(n_estimators = 100)
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)
CM = confusion_matrix(y_test, y_pred2)
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
TNR = TN/(TN+FP) # Specificity or true negative rate
FPR = FP/(FP+TN) # Fall out or false positive rate

print("Random Forest Clasifier")
print("Train Accuracy:",model2.score(x_train, y_train))
print("Test Accuracy:",model2.score(x_test, y_test))
print("Precision Score: ",metrics.precision_score(y_test, y_pred2))
print("Recall Score: ",metrics.recall_score(y_test, y_pred2)) #True positive rate
print("True Negative Rate: ", TNR)
print("False Positive Rate: ", FPR)
print("F2 Score: ",metrics.fbeta_score(y_test, y_pred2, beta=2.0))
print("F1 Score: ",metrics.f1_score(y_test, y_pred2))
print("Sensitivity: ",TP/(TP+FN))
print("Specificity: ",TNR)
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred2))
print("ROC curve ",metrics.plot_roc_curve(model2,x_test, y_test))
filename = 'RF_Model.pkl'
pickle.dump(model2, open(filename, 'wb'))


# In[61]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=0)
knn_clf=KNeighborsClassifier()
knn_clf.fit(x_train,y_train)
ypred=knn_clf.predict(x_test) #These are the predicted output values


# In[62]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, ypred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, ypred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,ypred)
print("Accuracy:",result2)


# ### 3)KNN

# In[53]:


from sklearn.neighbors import KNeighborsClassifier
model3 = KNeighborsClassifier(n_neighbors = 29)
model3.fit(x_train, y_train)
y_pred3 = model3.predict(x_test)
CM = confusion_matrix(y_test, y_pred3)
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
TNR = TN/(TN+FP) # Specificity or true negative rate
FPR = FP/(FP+TN)
print("KNN")
print("Train Accuracy:",model3.score(x_train, y_train))
print("Test Accuracy:",model3.score(x_test, y_test))
print("Precision Score: ",metrics.precision_score(y_test, y_pred3))
print("Recall Score: ",metrics.recall_score(y_test, y_pred3))
print("True Negative Rate: ", TNR)
print("False Positive Rate: ", FPR)
print("F2 Score: ",metrics.fbeta_score(y_test, y_pred3, beta=2.0))
print("F1 Score: ",metrics.f1_score(y_test, y_pred3))
print("Sensitivity: ",TP/(TP+FN))
print("Specificity: ",TNR)
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred3))
print("ROC curve ",metrics.plot_roc_curve(model3,x_test, y_test))
filename = 'KNN_Model.pkl'
pickle.dump(model3, open(filename, 'wb'))


# ### 4)NAIVE BAYES CLASSIFIER

# In[64]:


from sklearn.naive_bayes import GaussianNB
model4 = GaussianNB()
model4.fit(x_train, y_train)
y_pred4 = model4.predict(x_test)
CM = confusion_matrix(y_test, y_pred4)
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
TNR = TN/(TN+FP) # Specificity or true negative rate
FPR = FP/(FP+TN)
print("NAIVE BAYES CLASSIFIER")
print("Train Accuracy:",model4.score(x_train, y_train))
print("Test Accuracy:",model4.score(x_test, y_test))
print("Precision Score: ",metrics.precision_score(y_test, y_pred4))
print("Recall Score: ",metrics.recall_score(y_test, y_pred4))
print("True Negative Rate: ", TNR)
print("False Positive Rate: ", FPR)
print("F2 Score: ",metrics.fbeta_score(y_test, y_pred4, beta=2.0))
print("F1 Score: ",metrics.f1_score(y_test, y_pred4))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred4))
print("ROC curve ",metrics.plot_roc_curve(model4,x_test, y_test))
filename = 'NB_Model.pkl'
pickle.dump(model4, open(filename, 'wb'))


# ### 5)Logistic Regression

# In[65]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(multi_class='multinomial', random_state=1)
LR.fit(x_train, y_train)
y_LR = LR.predict(x_test)
CM = confusion_matrix(y_test, y_pred4)
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
TNR = TN/(TN+FP) # Specificity or true negative rate
FPR = FP/(FP+TN)
print("NAIVE BAYES CLASSIFIER")
print("Train Accuracy:",LR.score(x_train, y_train))
print("Test Accuracy:",LR.score(x_test, y_test))
print("Precision Score: ",metrics.precision_score(y_test, y_LR))
print("Recall Score: ",metrics.recall_score(y_test, y_LR))
print("True Negative Rate: ", TNR)
print("False Positive Rate: ", FPR)
print("F2 Score: ",metrics.fbeta_score(y_test, y_LR, beta=2.0))
print("F1 Score: ",metrics.f1_score(y_test, y_LR))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_LR))
print("ROC curve ",metrics.plot_roc_curve(LR,x_test, y_test))
filename = 'LR_Model.pkl'
pickle.dump(LR, open(filename, 'wb'))


# ### 6)SVM Linear kernel

# In[54]:


from sklearn import svm
model5 = svm.SVC(kernel='linear', probability=True)
model5.fit(x_train, y_train)
y_pred5 = model5.predict(x_test)
CM = confusion_matrix(y_test, y_pred5)
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
TNR = TN/(TN+FP) # Specificity or true negative rate
FPR = FP/(FP+TN)
print("SVM Linear kernel")
print("Train Accuracy:",model5.score(x_train, y_train))
print("Test Accuracy:",model5.score(x_test, y_test))
print("Precision Score: ",metrics.precision_score(y_test, y_pred5))
print("Recall Score: ",metrics.recall_score(y_test, y_pred5))
print("True Negative Rate: ", TNR)
print("False Positive Rate: ", FPR)
print("F2 Score: ",metrics.fbeta_score(y_test, y_pred5, beta=2.0))
print("F1 Score: ",metrics.f1_score(y_test, y_pred5))
print("Sensitivity: ",TP/(TP+FN))
print("Specificity: ",TNR)
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred5))
print("ROC curve ",metrics.plot_roc_curve(model5,x_test, y_test))
filename = 'SVML_Model.pkl'
pickle.dump(model5, open(filename, 'wb'))


# ### 7)SVM Polynomial

# In[55]:


model6 = svm.SVC(kernel='poly', degree=3, probability=True)
model6.fit(x_train, y_train)
y_pred6 = model6.predict(x_test)
CM = confusion_matrix(y_test, y_pred6)
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
TNR = TN/(TN+FP) # Specificity or true negative rate
FPR = FP/(FP+TN)
print("SVM Polynomial")
print("Train Accuracy:",model6.score(x_train, y_train))
print("Test Accuracy:",model6.score(x_test, y_test))
print("Precision Score: ",metrics.precision_score(y_test, y_pred6))
print("Recall Score: ",metrics.recall_score(y_test, y_pred6))
print("True Negative Rate: ", TNR)
print("False Positive Rate: ", FPR)
print("F2 Score: ",metrics.fbeta_score(y_test, y_pred6,beta=2.0))
print("F1 Score: ",metrics.f1_score(y_test, y_pred6))
print("Sensitivity: ",TP/(TP+FN))
print("Specificity: ",TNR)
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred6))
print("ROC curve ",metrics.plot_roc_curve(model6,x_test, y_test))
filename = 'SVMP_Model.pkl'
pickle.dump(model6, open(filename, 'wb'))


# ### 8)SVM sigmoid

# In[56]:


model7 = svm.SVC(kernel='sigmoid', probability=True)
model7.fit(x_train, y_train)
y_pred7 = model7.predict(x_test)
CM = confusion_matrix(y_test, y_pred7)
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
TNR = TN/(TN+FP) # Specificity or true negative rate
FPR = FP/(FP+TN)
print("SVM sigmoid")
print("Train Accuracy:",model7.score(x_train, y_train))
print("Test Accuracy:",model7.score(x_test, y_test))
print("Precision Score: ",metrics.precision_score(y_test, y_pred7))
print("Recall Score: ",metrics.recall_score(y_test, y_pred7))
print("True Negative Rate: ", TNR)
print("False Positive Rate: ", FPR)
print("F2 Score: ",metrics.fbeta_score(y_test, y_pred7, beta=2.0))
print("F1 Score: ",metrics.f1_score(y_test, y_pred7))
print("Sensitivity: ",TP/(TP+FN))
print("Specificity: ",TNR)
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred7))
print("ROC curve ",metrics.plot_roc_curve(model7,x_test, y_test))
filename = 'SVMS_Model.pkl'
pickle.dump(model7, open(filename, 'wb'))


# ### 9)SVM rbf

# In[57]:


model8 = svm.SVC(kernel='rbf', probability=True)
model8.fit(x_train, y_train)
y_pred8 = model8.predict(x_test)
CM = confusion_matrix(y_test, y_pred8)
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
TNR = TN/(TN+FP) # Specificity or true negative rate
FPR = FP/(FP+TN)
print("SVM rbf")
print("Train Accuracy:",model8.score(x_train, y_train))
print("Test Accuracy:",model8.score(x_test, y_test))
print("Precision Score: ",metrics.precision_score(y_test, y_pred8))
print("Recall Score: ",metrics.recall_score(y_test, y_pred8))
print("True Negative Rate: ", TNR)
print("False Positive Rate: ", FPR)
print("F2 Score: ",metrics.fbeta_score(y_test, y_pred8,beta=2.0))
print("F1 Score: ",metrics.f1_score(y_test, y_pred8))
print("Sensitivity: ",TP/(TP+FN))
print("Specificity: ",TNR)
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred8))
print("ROC curve ",metrics.plot_roc_curve(model8,x_test, y_test))
filename = 'SVMR_Model.pkl'
pickle.dump(model8, open(filename, 'wb'))


# ## Adaboost

# In[58]:


# Creating adaboost classifier model
from sklearn.ensemble import AdaBoostClassifier
import warnings
warnings.filterwarnings("ignore")

adb= AdaBoostClassifier()
model9 = adb.fit(x_train, y_train)
y_pred9 = model9.predict(x_test)
CM = confusion_matrix(y_test, y_pred9)
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
TNR = TN/(TN+FP) # Specificity or true negative rate
FPR = FP/(FP+TN)
print("Adaboost")
print("Train Accuracy:",model9.score(x_train, y_train))
print("Test Accuracy:",model9.score(x_test, y_test))
print("Precision Score: ",metrics.precision_score(y_test, y_pred9))
print("Recall Score: ",metrics.recall_score(y_test, y_pred9))
print("True Negative Rate: ", TNR)
print("False Positive Rate: ", FPR)
print("F2 Score: ",metrics.fbeta_score(y_test, y_pred9, beta=2.0))
print("F1 Score: ",metrics.f1_score(y_test, y_pred9))
print("Sensitivity: ",TP/(TP+FN))
print("Specificity: ",TNR)
print("Sensitivity: ",TP/(TP+FN))
print("Specificity: ",TNR)
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred9))
print("ROC curve ",metrics.plot_roc_curve(model9,x_test, y_test))
filename = 'Adaboost_Modelak.pkl'
pickle.dump(model9, open(filename, 'wb'))


# In[ ]:





# ## Single Image Detection

# In[70]:


#Recognition &Validation
#Assigning path with any any class image
data=[]
path=r"F:\Academic folders\TYSEM1\EDI\EDI_DATASET\Staircase\0_output.jpg"


#Repeated the process of image pre-processing and feature extraction
a=cv2.imread(path)
resize=(430,280)

#resize image
img=cv2.resize(a,resize)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#initialise sift descriptor
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

#convert the descriptor array into a dataframe format
out=pd.DataFrame(descriptors)
print("Descriptor Shape:",out.shape)

#initialise Kmeans and create 5 clusters


#train the model for the features i.e. for all elements in the Dataframe
array_double = np.array(out, dtype=np.double)
a=kmeans.predict(array_double)

#get the values of the histogram for one image only for 5 clusters i.e. in 5 bins
#kmeans.labels_ give us the label vlaue of the feature that its clustered into
#hist will give the hostogram for all those vlaues
hist=np.histogram(a,bins=[0,1,2,3,4,5,6])

#append the dataframe into the array in append mode, the array will only have 5 values which will store the values in a row
data.append(hist[0])

Output = pd.DataFrame(data)
print("Histogram:\n",Output)


# In[71]:


from sklearn.preprocessing import StandardScaler
MuktaS = StandardScaler()
Mukta = MuktaS.fit_transform(Output)


# In[72]:


Mukta


# In[73]:


Mukta.shape


# In[74]:


MB = pca.transform(Mukta)


# In[75]:


MB.shape


# In[76]:


MB


# In[77]:


pickle_in = open('KNN_Model.pkl', 'rb')
model1 = pickle.load(pickle_in )
pickle_in.close()


# In[78]:


#prediction
y_pred1 = model1.predict(MB)

#prints the prediction of the class
print(y_pred1)


# In[79]:


from sklearn.svm import SVC
DhoniSVM_LK = SVC(kernel='rbf')
Dhoni_LK = DhoniSVM_LK.fit(x_train, y_train)


# In[80]:


filename = 'SVC_RBF_PCA_3.sav'
pickle.dump(Dhoni_LK, open(filename, 'wb'))


# In[81]:


Dhoni_LK


# In[82]:


y_pred_LK = Dhoni_LK.predict(x_test)


# In[83]:


y_pred_LK


# In[84]:


from sklearn.metrics import confusion_matrix


# In[85]:


confusion_matrix(y_test, y_pred_LK)


# In[86]:


from sklearn.metrics import accuracy_score


# In[87]:


accuracy_score(y_test, y_pred_LK)


# In[88]:


y_pred1 = model1.predict(MB)

#prints the prediction of the class
print(y_pred1)


# # Classification using Voting classifiers

# In[79]:


from sklearn.naive_bayes import GaussianNB
# importing libraries
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# group / ensemble of models
estimator = []
estimator.append(('RFC',RandomForestClassifier(max_depth=13, n_estimators=170)))
estimator.append(('SVC', SVC(gamma ='auto', probability = True)))
estimator.append(('DTC', DecisionTreeClassifier()))
estimator.append(('KNN', KNeighborsClassifier(n_neighbors=30)))
estimator.append(('NBC', GaussianNB()))


# Voting Classifier with hard voting
vot_hard = VotingClassifier(estimators = estimator, voting ='hard')
vot_hard.fit(x_train, y_train)
y_pred = vot_hard.predict(x_test)
accuracy = vot_hard.score(x_test,y_test)
print(y_pred)
print(accuracy)

# group / ensemble of models
estimator = []
estimator.append(('RFC',RandomForestClassifier(max_depth=13, n_estimators=170)))
estimator.append(('SVC', SVC(gamma ='auto', probability = True)))
estimator.append(('DTC', DecisionTreeClassifier()))
estimator.append(('KNN', KNeighborsClassifier(n_neighbors=30)))
estimator.append(('NBC', GaussianNB()))


# Voting Classifier with soft voting
vot_soft = VotingClassifier(estimators = estimator, voting ='soft')
vot_soft.fit(x_train, y_train)
y_pred = vot_soft.predict(x_test)
accuracy = vot_soft.score(x_test,y_test)
print(y_pred)
print(accuracy)


# ## Testing And Validation of multiple Images

# In[80]:


import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import shutil
import random
import glob
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd              
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  
import PIL       
import PIL.Image
import os       
import os.path
from PIL import Image
import cv2 as cv
import cv2
from scipy.stats import stats
import pickle
from sklearn. model_selection import train_test_split
from sklearn.svm import SVC
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer   # to normalize sift feature extracted 


# In[82]:


pickle_in = open('KNN_Model.pkl', 'rb')
model = pickle.load(pickle_in )
pickle_in.close()


# In[83]:


pickle_in = open('Kmeans_CL_5_Model.sav', 'rb')
kmeans = pickle.load(pickle_in )
pickle_in.close()


# In[84]:


#load previously dumped hog image dataset 
pickle_in = open('PCA_3_Model.sav', 'rb')
pca = pickle.load(pickle_in )
pickle_in.close()


# In[90]:


b=r'F:\Academic folders\TYSEM1\EDI\EDI_DATASET\test'
data =[]
for filename in os.listdir(b):
    
    path=os.path.join(b,filename)
    a=cv2.imread(path)
    
    #resize imageAC
    resize=(512,512)
    img=cv2.resize(a,resize)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #initialise sift descriptor
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    #convert the descriptor array into a dataframe format
    out=pd.DataFrame(descriptors)
    array_double = np.array(out, dtype=np.double)
#     print(out)

    a=kmeans.predict(array_double) 
    hist=np.histogram(a,bins=[0,1,2,3,4,5,6])
    data.append(hist[0])
    Output = pd.DataFrame(data)
    #print("Histogram:\n",Output)

norm = Normalizer()
normalized = norm.fit_transform(Output)  
normalized.shape# normalize 

Standardize = StandardScaler()
x_scaled = Standardize.fit_transform(normalized)
pd.DataFrame(normalized)
print(x_scaled.shape)


x_pca = pca.transform(x_scaled)
x_pca = pd.DataFrame(x_pca)
x_pca

# First 1000 images are of traffic
#m = x_pca.iloc[:1000,:]
#prediction
y_pred1 = model.predict(x_pca)

#prints the prediction of the class
print(y_pred1)
y_pred1
a=np.count_nonzero(y_pred1==1)
print("Traffic :",a)
a=np.count_nonzero(y_pred1==0)
print("Non Traffic :",a)


# In[ ]:




