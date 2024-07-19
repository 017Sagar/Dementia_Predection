from django.shortcuts import render
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

def Machine_learning(request):
    if request.method=="POST":
        data=request.POST
        
        vi=data.get("testvist")
        mr=data.get("testmr")
        mf=data.get("testmf")
        age=data.get("testage")
        educ=data.get("testeduc")
        ses=data.get("testses")
        mmse=data.get("testmmse")
        cdr=data.get("testcdr")
        ctiv=data.get("testctiv")
        mwbv=data.get("testmwbv")
        asf=data.get("testasf")
        ih=data.get("testash")
        io=data.get("testaso")

#         df=pd.read_csv("C:\\Users\\SAGAR M S\\final_project_ML\\dementia_dataset.csv")
#         df['SES'].fillna(df['SES'].median(),inplace=True)
#         df['MMSE'].fillna(df['MMSE'].median(),inplace=True)
#         df = df.drop('Hand', axis=1)
#         df = df.drop('Subject ID', axis=1)
#         df = df.drop('MRI ID', axis=1)
#         d = {'M': 0, 'F': 1}
#         df['M/F'] = df['M/F'].map(d).fillna(df['M/F'])
#         d = {'Nondemented': 0, 'Demented': 1,'Converted':2}
#         df['Group'] = df['Group'].map(d).fillna(df['Group'])

#         X = df.drop(['Group'], axis='columns')
#         y = df.Group

#         from sklearn.preprocessing import MinMaxScaler

#         scaler = MinMaxScaler()

# # scaler.fit(df[['Visit','MR Delay','M/F','Age','EDUC','SES','MMSE','CDR','eTIV','nWBV','ASF']])
# # df['Visit','MR Delay','M/F','Age','EDUC','SES','MMSE','CDR','eTIV','nWBV','ASF'] = scaler.transform(df[['Visit','MR Delay','M/F','Age','EDUC','SES','MMSE','CDR','eTIV','nWBV','ASF']])

#         scaler.fit(df[['Group']])
#         df['Group'] = scaler.transform(df[['Group']])


#         scaler.fit(df[['MR Delay']])
#         df['MR Delay'] = scaler.transform(df[['MR Delay']])


#         scaler.fit(df[['M/F']])
#         df['M/F'] = scaler.transform(df[['M/F']])



#         scaler.fit(df[['Age']])
#         df['Age'] = scaler.transform(df[['Age']])



#         scaler.fit(df[['EDUC']])
#         df['EDUC'] = scaler.transform(df[['EDUC']])



#         scaler.fit(df[['SES']])
#         df['SES'] = scaler.transform(df[['SES']])


#         scaler.fit(df[['MMSE']])
#         df['MMSE'] = scaler.transform(df[['MMSE']])


#         scaler.fit(df[['CDR']])
#         df['CDR'] = scaler.transform(df[['CDR']])


#         scaler.fit(df[['eTIV']])
#         df['Group'] = scaler.transform(df[['eTIV']])


#         scaler.fit(df[['Group']])
#         df['eTIV'] = scaler.transform(df[['eTIV']])



#         scaler.fit(df[['nWBV']])
#         df['nWBV'] = scaler.transform(df[['nWBV']])


#         scaler.fit(df[['ASF']])
#         df['ASF'] = scaler.transform(df[['ASF']])

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#         sc=StandardScaler()
#         X_train = sc.fit_transform(X_train)
#         X_test= sc.transform(X_test)

#         model = SVC()
#         model.fit(X_train, y_train)
        heart_data = pd.read_csv("C:\\Users\\SAGAR M S\\final_project_ML\\heart_disease_data.csv")

        heart_data['target'].value_counts()
        X = heart_data.drop(columns='target', axis=1)
        Y = heart_data['target']
        print(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
        print(X.shape, X_train.shape, X_test.shape)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = SVC(kernel='linear', random_state=2)
        # training the LogisticRegression model with Training data
        model.fit(X_train, Y_train) 

        result=model.predict([[float(vi),float(mr),float(mf),float(age),float(educ),float(ses),float(mmse),float(cdr),float(ctiv),float(mwbv),float(asf),float(ih),float(io)]])

        print(result[0])
        return render(request,'Machine_learning.html',context={'result':result[0]})

    return render(request,'Machine_learning.html')
