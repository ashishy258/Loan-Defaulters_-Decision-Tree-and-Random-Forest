


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data



df=pd.read_csv('loan_data.csv')




df.head()



df['not.fully.paid'].value_counts()



df['fico'].hist(bins=30,color='darkred',alpha=0.7)




sns.set_style('whitegrid')
sns.countplot(x='fico',hue='not.fully.paid',data=df,palette='RdBu_r')


# ** Create a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid. **



sns.set_style('whitegrid')
sns.countplot(x='purpose',hue='not.fully.paid',data=df,palette='RdBu_r')






# ## Categorical Features
# 
# Notice that the **purpose** column as categorical
# 
# That means we need to transform them using dummy variables so sklearn will be able to understand them. Let's do this in one clean step using pd.get_dummies.
# 
# Let's show you a way of dealing with these columns that can be expanded to multiple categorical features if necessary.
# 
# **Create a list of 1 element containing the string 'purpose'. Call this list cat_feats.**



cat_feats=['purpose']




final_data = pd.get_dummies(df,columns=cat_feats,drop_first=True)




final_data.head()


# ## Train Test Split



X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)




from sklearn.tree import DecisionTreeClassifier



dtree = DecisionTreeClassifier()




dtree.fit(X_train,y_train)




predictions = dtree.predict(X_test)




from sklearn.metrics import classification_report,confusion_matrix




print(classification_report(y_test,predictions))



print(confusion_matrix(y_test,predictions))


# ## Training the Random Forest model
# 
# Now its time to train our model!
# 
# **Create an instance of the RandomForestClassifier class and fit it to our training data from the previous step.**



from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)




rfc_pred = rfc.predict(X_test)



rfc_pred = rfc.predict(X_test)




print(confusion_matrix(y_test,rfc_pred))



print(classification_report(y_test,rfc_pred))