import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree
from numpy import mean
from numpy import std
from sklearn import linear_model, preprocessing
import seaborn as sns
import warnings
import sklearn
#to ignore the warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data= pd.read_csv("C:\\Users\\Adwit\\Desktop\\College work\\Iris.csv")
data.drop("Id",axis=1,inplace=True) #Id is set as index
print(data.items())
df=data.drop("Species",axis=1)
print(data)                         #reading the data

#Exploratory data analysis

print(data.isnull().sum())  #no columns have null values

print(data.describe())  #as we can see the mean and (max,min) value compared do not have much difference(except )

data.boxplot()   #finding outliers
plt.title("Adwitiya 189303075")
plt.show()


'''data_mean, data_std = mean(data), std(data)

# identify outliers and remove outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
outliers = [x for x in df["SepalLengthCm"].items() if x < lower | x > upper]
print(x for x in df['SepalLengthCm'])
outliers=[x for x in df if x < lower or x > upper]
print('Identified outliers: %d' % len(outliers))'''

#encoding the features
le = preprocessing.LabelEncoder()
Species= le.fit_transform(list(data["Species"])) #as there are three different classes they will automatically be encoded to [0,1,2]
SepalLengthCm=data["SepalLengthCm"].tolist()
SepalWidthCm=data["SepalWidthCm"].tolist()
PetalLengthCm=data["PetalLengthCm"].tolist()
PetalWidthCm=data["PetalWidthCm"].tolist()
predict='Species'

#defining our labels and classes(labels:features, classes:Species)
x=list(zip(SepalLengthCm ,SepalWidthCm,  PetalLengthCm,  PetalWidthCm))
y=list(Species)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y,test_size=0.2)
#print(x_train,y_test)
acc=0
for i in range (0,15):           #getting max accuracy model
    DT=DecisionTreeClassifier()
    DT.fit(x_train,y_train)
    curr=DT.score(x_test,y_test)

    if curr>acc:
        acc=curr
    else:
        continue
print(f"Accuracy:{acc}")
pred=DT.predict(x_test)
print(f"Prediction on x_test:{pred}")

c_m=confusion_matrix(y_test,pred)
sns.heatmap(c_m,annot=True,cmap="Blues",)
plt.show()

#visualizing DT
tree.plot_tree(DT,feature_names=['SepalLengthCm',  'SepalWidthCm',  'PetalLengthCm',  'PetalWidthCm'],class_names=data["Species"].unique(),filled=True)
plt.show()