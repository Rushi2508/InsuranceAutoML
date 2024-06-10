from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import numpy as np 
from sklearn.linear_model import LogisticRegression
import requests
import xgboost as xgb
from sklearn.metrics import mean_squared_error 
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)


#defining a list 
dev = []
data = []
# Load the dataset
df = pd.read_csv("C:\\Users\\YutishthaTanwar\\Downloads\\Insurance_Final.csv")
#checking null value
print(df.isnull().sum())

# Convert 'DateTime' column to DateTimetime object
#df['DateTime'] = pd.to_datetime(df['DateTime'], format='%m-%d-%Y')

# Now drop the 'DateTime' column
#df = df.drop(columns=['DateTime'])

# Define categorical columns for one-hot encoding
categorical_cols = ['Claimed','Policy Type']

# Drop the target variable 'Production Cost' from the encoded DataFrame to get features (X)
# Perform one-hot encoding
encoded_X = pd.get_dummies(df, columns=categorical_cols)

X = encoded_X.drop(columns=['Amount','DateTime'])

# Assign the target variable 'Production Cost' to y
y = df['Amount']

#standard scaling 
scaler = StandardScaler() 
X = scaler.fit_transform(X) 

# Splitting dataset into train and test set 
X_train, X_test, Y_train, Y_test = train_test_split( X, y, test_size=0.3, random_state=0) 
  
#----------------------LinearRegression-----------------------------

regression_model = LinearRegression().fit(X_test, Y_test)
regression_Y_pred = regression_model.predict(X_test) 
dev.append(regression_model.score(X_test, Y_test))
data.append(regression_Y_pred)

#----------------------LassoRegression------------------------------

# Model training 
lasso_model = Lasso(alpha=0.01)  # You can adjust the alpha parameter as needed
lasso_model.fit(X_train, Y_train)  
# Prediction on test set using Lasso model
lasso_Y_pred = lasso_model.predict(X_test) 
# Print the R-squared score of the Lasso model
dev.append(lasso_model.score(X_test, Y_test)) 
data.append(lasso_Y_pred)

#----------------------RidgeRegression------------------------------

clf = Ridge(alpha=1.0)
clf.fit(X_train, Y_train)
clf_Y_Pred = clf.predict(X_test)
dev.append(clf.score(X_test, Y_test))
data.append(clf_Y_Pred)

#----------------------LogisticRegression---------------------------

clf1 = LogisticRegression(random_state=0)
clf1.fit(X_train, Y_train)
yx_pred = clf1.predict(X_test)
dev.append(clf1.score(X_test, Y_test))
data.append(yx_pred)

#----------------------ScoreDisplay--------------------------------

print("this is data shape: " + str(len(data[3])))

#----------------------MaxPerformanceIndex-------------------------

ind = 0    #ind variable to store the index of maximum value in the list
max_element = dev[0]
 
for i in range (1,len(dev)): #iterate over array
  if dev[i] > max_element: #to check max value
    max_element = dev[i]
    ind = i

#----------------------StoringPredictions-------------------------

# Create a new DataFrame to store predictions
predictions_df = pd.DataFrame({'prediction': data[ind]})

# Concatenate predictions_df with the original DataFrame df
df_with_predictions = pd.concat([df, predictions_df], axis=1)
df = df_with_predictions.merge(df_with_predictions[['prediction']], how='left', left_index=True, right_index=True)

print("successfully added")

#----------------------TimeSeriesAnalysis-------------------------


#df = pd.read_csv(r"C:\Users\YutishthaTanwar\Downloads\PJME_hourly (1).csv")
df = df.reset_index()
df = df.set_index('DateTime')
df.index = pd.to_datetime(df.index)

train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2015']

def create_features(df):
    
    #Create time series features based on time series index.
    
    df= df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = create_features(df)


train = create_features(train)
test = create_features(test)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
TARGET = ['Amount']

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)

fi = pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importance'])


test['TSAprediction'] = reg.predict(X_test)
if 'Yes' in test['Claimed'].values:
       df = df.merge(test[['TSAprediction']], how='left', left_index=True, right_index=True)
print("predictions are printed here" + str(test['TSAprediction']))

#----------------------SavingDFToCSV-------------------------
df.to_csv('C:\\Users\\YutishthaTanwar\\Downloads\\InsuranceML_predictions.csv', index=False)

print("successfully added")
#plt.plot(df['prediction_x'],test['TSAprediction'])'''

#----------------------SendingDataToQLik--------------------------

@app.route('/receive_csv', methods=['POST'])
def receive_csv():
    uploaded_file = request.files['InsuranceML_predictions.csv']
    uploaded_file.save('uploaded_file.csv')
    url = 'https://your-qlik-server.com/your-connector-endpoint'
    files = {'file': open('uploaded_file.csv', 'rb')}
    response = requests.post(url, files=files)
    if response.ok:
            return jsonify({'message': 'CSV file sent to Qlik Sense successfully'})
    else:
            return jsonify({'error': 'Failed to send CSV file to Qlik Sense'})

if __name__ == '__main__':
    app.run(debug=True)
    
'''
# Convert the DataFrame to JSON format
data_to_send = df.to_json(orient='records')

# Authentication: Replace 'YOUR_ACCESS_TOKEN' with your actual access token
access_token = 'eyJhbGciOiJFUzM4NCIsImtpZCI6IjI5N2MzZTk0LWMwY2YtNDBhZC1hYTQ2LTMwNTQxNjk3MDVhYiIsInR5cCI6IkpXVCJ9.eyJzdWJUeXBlIjoidXNlciIsInRlbmFudElkIjoiQTBMRXdJbXRFbU45b0ZKb2pSaVlKc0RGZ3h0TnlHRGMiLCJqdGkiOiIyOTdjM2U5NC1jMGNmLTQwYWQtYWE0Ni0zMDU0MTY5NzA1YWIiLCJhdWQiOiJxbGlrLmFwaSIsImlzcyI6InFsaWsuYXBpL2FwaS1rZXlzIiwic3ViIjoiNjRjOGI4M2RhN2Y2YjI4YmY3OTg3ZDI5In0.US5fT0cNKVqalrnh4HA_RLOboIYYN4lfzlG64arjyTgl6-CwBUH54l1K03wpHZZ91qc6BLxs4-OAmchCwAHkojiC2Uju22I8C9Z_-uTOrtO5CUGI79tMBm4qYpH2Wjwq'
# API endpoint for sending data to Qlik Sense
url = 'https://bdoindiacoe.sg.qlikcloud.com/sense/app/ece9f334-ad23-4c0c-8e72-0fc2d4c2656a'

# Make HTTP POST request
headers = {
    'Authorization': f'Bearer {access_token}',
    'Content-Type': 'application/json'
}

response = requests.post(url, json=data_to_send, headers=headers)
# Process response
if response.ok:
    print('Data sent successfully to Qlik Sense!')
else:
    print(f'Error: {response.status_code} - {response.text}')
'''