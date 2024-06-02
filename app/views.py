from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier

def index(request):
    return render(request,'home.html')

def proceed(request):
    global data
    if request.method=='POST':
        file = request.FILES['filename']
        data=pd.read_csv(file)
        data['bmi'].fillna(data['bmi'].median(), inplace=True)
        data.drop('id', axis=1, inplace=True)
        data['gender'].replace(['Male', 'Female', 'Other'], [0, 1, 2], inplace=True)
        data['ever_married'].replace(['Yes', 'No'], [0, 1], inplace=True)
        data['work_type'].replace(['Private', 'Self-employed', 'children', 'Govt_job', 'Never_worked'], [0, 1, 2, 3, 4],inplace=True)
        data['Residence_type'].replace(['Urban', 'Rural'], [0, 1], inplace=True)
        data['smoking_status'].replace(['never smoked', 'Unknown', 'formerly smoked', 'smokes'], [0, 1, 2, 3],inplace=True)
        for i in ['avg_glucose_level']:
            q3, q1 = np.percentile(data.loc[:, i], [75, 25])
            IQR = q3 - q1
            max = q3 + (1.5 * IQR)
            min = q1 - (1.5 * IQR)
            data.loc[data[i] < min, i] = np.nan
            data.loc[data[i] > max, i] = np.nan
            data['avg_glucose_level'].fillna(data['avg_glucose_level'].median(), inplace=True)
        for i in ['bmi']:
            q3, q1 = np.percentile(data.loc[:, i], [75, 25])
            IQR = q3 - q1
            max = q3 + (1.5 * IQR)
            min = q1 - (1.5 * IQR)
            data.loc[data[i] < min, i] = np.nan
            data.loc[data[i] > max, i] = np.nan
            data['bmi'].fillna(data['bmi'].median(), inplace=True)

            col = data.columns
            rows=data.values.tolist()
            return render(request,'showdata.html',{'cols':col,'rows':rows})
    return render(request,'index.html')

def modelselection(request):
    global X_train, X_test, y_train, y_test,model_h,auc,rfc_acc,accuracy,acc
    if request.method=='POST':
        X = data.drop('stroke', axis=1)
        y = data['stroke']
        os = SMOTE()
        X, y = os.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model=request.POST['models']
        if model == '1':
            model_n = DecisionTreeClassifier()
            model_n.fit(X_train, y_train)
            pred = model_n.predict(X_test)
            auc = accuracy_score(pred, y_test)
            msg = 'The accuracy of DecisionTreeClassifier :' + str(auc)
            return render(request,'modelselection.html',{'msg':msg})
        elif model == '2':
            rfc = RandomForestClassifier(n_estimators=90)
            rfc.fit(X_train, y_train)
            pred1 = rfc.predict(X_test)
            rfc_acc = accuracy_score(pred1, y_test)
            msg = 'The accuracy of Random Forest Classifier :' + str(rfc_acc)
            return render(request, 'modelselection.html', {'msg': msg})
        elif model=='3':
            model = XGBClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_pred, y_test)
            msg = 'The accuracy of XGBoost Classifer :' + str(accuracy)
            return render(request, 'modelselection.html', {'msg': msg})
        elif model == '4':
            level0 = list()
            level0.append(('DT', DecisionTreeClassifier()))
            level0.append(('XGBoost', XGBClassifier()))
            level0.append(('RF', RandomForestClassifier(n_estimators=200)))
            model_h=StackingClassifier(estimators=level0)
            model_h.fit(X_train, y_train)
            y = model_h.predict(X_test)
            acc = accuracy_score(y_test, y)
            msg = 'The accuracy of Hybrid Model :' +str(acc)
            return render(request, 'modelselection.html', {'msg':msg})
       
        else:
            print("hellooooooooooooooo")
    return render(request,'modelselection.html')


def graph(request):
    # Assuming you have the accuracies stored in variables auc, rfc_acc, acc, accuracy
    x = ['Decision Tree', 'Random Forest', 'Hybrid Model', 'XGB']
    y = [auc, rfc_acc, acc, accuracy]

    data = {'Models': x, 'Accuracy': y}
    graph = sns.barplot(x='Models', y='Accuracy', data=data)

    plt.title('Model Accuracies')
    plt.ylabel('Accuracy')
    plt.xlabel('Models')

    # Add accuracy values on top of the bars
    for i, v in enumerate(y):
        graph.text(i, v + 0.01, f'{np.round(v, 2)*100}%', ha='center', va='bottom')

    plt.show()

    return render(request, 'predict_val.html')

def predict(request):
    global rfc
    from sklearn.calibration import CalibratedClassifierCV
    if request.method == 'POST':
        gender = request.POST['gender']
        age = request.POST['age']
        hypertension = request.POST['hypertension']
        heart_disease = request.POST['heart_disease']
        ever_married = request.POST['ever_married']
        work_type = request.POST['work_type']
        Residence_type = request.POST['Residence_type']
        avg_glucose_level = request.POST['avg_glucose_level']
        bmi = request.POST['bmi']
        smoking_status = request.POST['smoking_status']

        
        input_values = {
            'gender': gender,'age': age,'hypertension': hypertension,'heart_disease': heart_disease,'ever_married': ever_married,'work_type': work_type,'Residence_type': Residence_type,'avg_glucose_level': avg_glucose_level,'bmi': bmi,'smoking_status': smoking_status
        }

        x = [[float(gender), float(age), float(hypertension), float(heart_disease), float(ever_married),
              float(work_type), float(Residence_type), float(avg_glucose_level), float(bmi), float(smoking_status)]]

        y = pd.DataFrame(x, columns=X_train.columns)
        y_pred = model_h.predict(y)

        if y_pred == [1]:
            messages = 'There is a chance to get stroke'
            
            
            probability = y_pred[0] if y_pred.ndim > 1 else y_pred[0]

            
            plt.figure(figsize=(8, 6))
            plt.bar(['No Heart Disease', 'Heart Disease'], [1 - probability, probability], color=['green', 'red'])
            plt.title('Probability of Heart Disease Prediction')
            plt.xlabel('Prediction')
            plt.ylabel('Probability')
            plt.show()
            
        else:
            messages = 'There is no chance to get stroke'
           
            probability = y_pred[0] if y_pred.ndim > 1 else y_pred[0]

            # Plot the probability graph
            plt.figure(figsize=(8, 6))
            plt.bar(['No Heart Disease', 'Heart Disease'], [1 - probability, probability], color=['green', 'red'])
            plt.title('Probability of Heart Disease Prediction')
            plt.xlabel('Prediction')
            plt.ylabel('Probability')
            plt.show()
           
            # dc = Mypridiction.objects.create(gender=gender,age= age,hypertension=hypertension,heart_disease =heart_disease,ever_married =ever_married,work_type= work_type,Residence_type= Residence_type,avg_glucose_level= avg_glucose_level,bmi= bmi,smoking_status= smoking_status,
                
                
            #     output=messages)

       
        return render(request, 'predict_val.html', {'input_values': input_values, 'result_message': messages})

    return render(request, 'predict_val.html')
