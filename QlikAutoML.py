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
from flask import Flask, request, jsonify # type: ignore
import requests
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

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



#----------------------ScoreDisplay--------------------------------

#print("this is data shape: " + str(len(data[3])))

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

#----------------------Classification-------------------------
dev1 =[]
data1=[]
categorical_cols1 = ['Amount','Policy Type']

# Drop the target variable 'Production Cost' from the encoded DataFrame to get features (X)
# Perform one-hot encoding
encoded_X1 = pd.get_dummies(df, columns=categorical_cols1)

X_cat = encoded_X1.drop(columns=['Claimed'])

# Assign the target variable 'Production Cost' to y
y_cat = df['Claimed']

#standard scaling 
scaler1 = StandardScaler() 
X_cat = scaler1.fit_transform(X_cat) 

# Splitting dataset into train and test set 
X_train_cat, X_test_cat, Y_train_cat, Y_test_cat = train_test_split( X_cat, y_cat, test_size=0.3, random_state=0)

# Initialize SimpleImputer
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training data and transform both training and testing data
X_train_cat_imputed = imputer.fit_transform(X_train_cat)
X_test_cat_imputed = imputer.transform(X_test_cat)
#----------------------LogisticRegression---------------------------

clf11 = LogisticRegression(random_state=0)
clf11.fit(X_train_cat_imputed, Y_train_cat)
yx_pred1 = clf11.predict(X_test_cat_imputed)
dev1.append(clf11.score(X_test_cat_imputed, Y_test_cat))
data1.append(yx_pred1)

#----------------------DecisionTree---------------------------

clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit(X_train_cat_imputed, Y_train_cat)
yx_pred11 = clf1.predict(X_test_cat_imputed)
dev1.append(clf1.score(X_test_cat_imputed, Y_test_cat))
data1.append(yx_pred11)

#----------------------RandomForest---------------------------

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train_cat_imputed, Y_train_cat)
yxx_pred = clf.predict(X_test_cat_imputed)
dev1.append(clf.score(X_test_cat_imputed, Y_test_cat))
data1.append(yxx_pred)

#----------------------SVM------------------------------------
# Define the pipeline with an imputer and SVC
pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),  # Impute missing values with the mean
    StandardScaler(),  # Standardize features by removing the mean and scaling to unit variance
    svm.SVC()  # Support Vector Classifier
)

# Fit the pipeline on the data
pipeline.fit(X_train_cat_imputed, Y_train_cat)

yxxx_pred = pipeline.predict(X_test_cat_imputed)
dev1.append(pipeline.score(X_test_cat_imputed, Y_test_cat))
data1.append(yxxx_pred)

#----------------------MaxPerformanceIndex-------------------------

ind1 = 0    #ind variable to store the index of maximum value in the list
max_element1 = dev1[0]
 
for i in range (1,len(dev1)): #iterate over array
  if dev1[i] > max_element1: #to check max value
    max_element1 = dev1[i]
    ind1 = i

#----------------------StoringPredictions-------------------------

# Create a new DataFrame to store predictions
#predictions_df1 = pd.DataFrame({'claimprediction': data1[ind1]}, index=df.index[X_test_cat_imputed])
#predictions_df1 = pd.DataFrame({'claimprediction': data1[ind1]}, index=range(len(X_test_cat_imputed)))
#predictions_df1 = pd.DataFrame({'claimprediction': data1[ind1]}, index=df.index)
# Concatenate predictions_df with the original DataFrame df
selected_predictions = data1[ind1]




# Check the length of predictions and the index of the original DataFrame
print("Length of predictions:", len(selected_predictions))
print("Length of DataFrame index:", len(df.index))


#----------------------StoringPredictions-------------------------

# Create a new DataFrame to store predictions
#predictions_df1 = pd.DataFrame({'claimprediction': selected_predictions}, index=df.index)
#predictions_df1 = pd.DataFrame({'claimprediction': selected_predictions}, index=df.index)
# Create a new DataFrame with predictions aligned with DataFrame index
predictions_df1 = pd.DataFrame({'claimprediction': selected_predictions}, index=df.index[:len(selected_predictions)])

# Check the length of the predictions DataFrame
print("Length of predictions DataFrame:", len(predictions_df1))
df_with_predictions1 = pd.concat([df, predictions_df1], axis=1)
df_with_predictions1.to_csv('C:\\Users\\YutishthaTanwar\\Downloads\\InsuranceML_predictions.csv', index=False)

print("successfully added")





'''
import csv
import psycopg2

# Establish connection to PostgreSQL
conn = psycopg2.connect(
    dbname="your_database",
    user="your_username",
    password="your_password",
    host="localhost",
    port="5432"
)

# Open CSV file
with open('your_csv_file.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip header row if exists

    # Create a cursor object
    cursor = conn.cursor()

    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Define the SQL statement to insert data into your table
        insert_query = """
        INSERT INTO your_table_name (column1, column2, column3, ...) 
        VALUES (%s, %s, %s, ...);
        """

        # Execute the SQL statement with the current row's values
        cursor.execute(insert_query, row)

# Commit the transaction
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()

print("CSV data has been successfully inserted into PostgreSQL.")

#plt.plot(df['prediction_x'],test['TSAprediction'])
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
    app.run(port = 5000, debug=True)
    ngrok http 5000




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