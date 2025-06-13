# Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.filterwarnings('ignore')
file_path = r"C:\Users\NEW\OneDrive\CA2 AI\CVD_cleaned.csv"
data = pd.read_csv(file_path)
# print(data)
# # print(df.isnull().sum())
data1 = data.head(1000)
# print(data1)
# print(data1['Age_Category'])
df = data1.drop(columns=['Fruit_Consumption','Green_Vegetables_Consumption','FriedPotato_Consumption', 'Arthritis','Skin_Cancer', 'Other_Cancer', 'Height_(cm)', 'BMI', 'Checkup'])
# print(df.head(20))
age_mapping = {
    '18-24': 0,
    '25-29': 1,
    '30-34': 2,
    '35-39': 3,
    '40-44': 4,
    '45-49': 5,
    '50-54': 6,
    '55-59': 7,
    '60-64': 8,
    '65-69': 9,
    '70-74': 10,
    '75-79': 11,
    '80+': 12
}
df['Age_Category'] = df['Age_Category'].map(age_mapping)

label_mapping = {
    'Poor': 0,
    'Fair': 1,
    'Good': 2,
    'Very Good': 3,
    'Excellent': 4
}

# Apply the mapping to the health status column
df['General_Health'] = df['General_Health'].map(label_mapping)

diabetes_mapping = {
    'No': 0,
    'Yes': 1,
    'No, pre-diabetes or borderline diabetes': 2
}

df['Diabetes'] = df['Diabetes'].map(diabetes_mapping)

df['Alcohol_Consumption_Cat'] = df['Alcohol_Consumption'].apply(lambda x: 'Yes' if x > 0 else 'No')
# Label encoding 'Yes' as 1 and 'No' as 0
df['Alcohol_Consumption_Cat'] = df['Alcohol_Consumption_Cat'].map({'Yes': 1, 'No': 0})


from sklearn.preprocessing import LabelEncoder

columns_to_encode = ['General_Health','Exercise', 'Heart_Disease', 'Depression', 'Diabetes', 'Sex', 'Age_Category', 'Smoking_History']  # Update with your columns

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to each column that needs encoding
for col in columns_to_encode:
    df[col] = label_encoder.fit_transform(df[col])

# Display the DataFrame after encoding
# print(df.head(40))
# print(df.dtypes)

corr_matrix = df.corr()

# Create a heatmap using seaborn
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Show the heatmap
plt.title('Correlation Heatmap')
# plt.show()


X = df.drop(['Heart_Disease','Alcohol_Consumption'],axis=1)
y = df.Heart_Disease
# print(X)
# print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y , train_size = 0.8 , random_state = 1)

#Logistic Regresson
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression(class_weight='balanced')

logreg.fit(X_train , y_train)
y_pred = logreg.predict(X_test)
print("Logistic Regresson Accuracy is :", logreg.score(X_test , y_test))

from sklearn.metrics import classification_report

predictions = logreg.predict(X_test)
print(classification_report(y_test, predictions))

# General_Health = int(input("Enter General Health(e.g.Poor: 0, Fair: 1, Good: 2, Very Good: 3, Excellent: 4): "))
# Exercise = int(input("Enter Exercise (e.g.,1 for Yes, 0 for No): "))
# Depression = int(input("Enter Depression (e.g., No: 0, Yes: 1): "))
# Diabetes = int(input("Enter Diabetes (e.g., No: 0, Yes: 1, No, pre-diabetes or borderline diabetes: 2): "))
# Sex = int(input("Enter Sex (0 for Male, 1 for Female): "))
# Age_Category = int(input("Enter Age Category (e.g., 18-24: 0, 25-29': 1,30-34: 2, 35-39: 3, 40-44: 4, 45-49: 5, 50-54: 6, 55-59: 7, 60-64: 8, 65-69: 9, 70-74: 10, 75-79: 11, 80+: 12): "))  # Example: 10 corresponds to 70-74
# Weight = float(input("Enter Weight in kg (e.g., 32.66): "))
# Smoking_History = int(input("Enter Smoking History (1 for Yes, 0 for No): "))
# Alcohol_Consumption = int(input("Enter Alcohol Consumption (1 for Yes, 0 for No): "))
# user_input = [[General_Health, Exercise, Depression, Diabetes, Sex, Age_Category, Weight, Smoking_History, Alcohol_Consumption]]

# example = logreg.predict(user_input)
# print(example)
# if example[0] == 1:
#     print("Heart disease is likely to happen in the future.")
# else:
#     print("Heart disease is not likely to happen in the future.")


from flask import Flask, render_template, request, jsonify
import pickle

# # Load your logistic regression model (Ensure you have a saved model file)
with open('logreg_model.pkl', 'wb') as file:
    pickle.dump(logreg, file)
model_path = 'logreg_model.pkl'
with open('logreg_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    # Render the form
    return render_template('ui.html')

@app.route('/predict', methods=['POST'])
def predict():
        data = request.form
    
        General_Health = int(data['General_Health'])
        Exercise = int(data['Exercise'])
        Depression = int(data['Depression'])
        Diabetes = int(data['Diabetes'])
        Sex = int(data['Sex'])
        Age_Category = int(data['Age_Category'])
        Weight = float(data['Weight'])
        Smoking_History = int(data['Smoking_History'])
        Alcohol_Consumption_Cat = int(data['Alcohol_Consumption'])

        # Prepare the input for the model
        user_input = np.array([[General_Health, Exercise, Depression, Diabetes, Sex, Age_Category, Weight, Smoking_History, Alcohol_Consumption_Cat]])

        # Predict the output
        prediction = logreg.predict(user_input)
      
        # Create a message based on prediction
        result = "You are at risk of heart disease." if prediction == 1 else "You are not at risk of heart disease."

        return render_template('ui.html', prediction_text='Prediction: {}'.format(result))

if __name__ == '__main__':
    app.run(debug=True)

# print("numpy", np.__version__)
# print("pandas", pd.__version__)
# print("sklearn", sklearn.__version__)