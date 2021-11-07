import pandas as pd
import numpy as np
import datetime as dt
import plotly.offline as py
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# run block of code and catch warnings
import warnings
with warnings.catch_warnings():
	# ignore all caught warnings
	warnings.filterwarnings("ignore")

import pickle
import streamlit as st
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split

# loading in the model to predict on the data
pickle_in = open('output.pkl', 'rb')
rand_clf = pickle.load(pickle_in)

df = pd.read_csv('insurance_claims.csv')
df = df.rename(columns={'capital-gains': 'capital_gains', 'capital-loss': 'capital_loss'})
# removing column named _c39 as it contains only null values
df = df.drop(['_c39'], axis = 1)
df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
drop_columns = ['policy_state', 'policy_csl', 'incident_date', 'incident_state', 'incident_city', 'incident_location']
df = df.drop(drop_columns, axis = 1)
df['fraud_reported'] = df['fraud_reported'].str.replace('Y', '1')
df['fraud_reported'] = df['fraud_reported'].str.replace('N', '0')
df['fraud_reported'] = df['fraud_reported'].astype(int)

Fraud = df[df['fraud_reported'] == 1]
Valid = df[df['fraud_reported'] == 0]


# defining the function which will make the prediction using
# the data which the user inputs

def vis_data(df, x, y = 'fraud_reported', graph = 'countplot'):
    if graph == 'hist':
        fig = px.histogram(df, x = x)
        fig.update_layout(title = 'Distribution of {x}'.format(x = x))
        st.write(fig)
    elif graph == 'bar':
      fig = px.bar(df, x = x, y = y)
      fig.update_layout(title = '{x} vs. {y}'.format(x = x, y = y))
      st.write(fig)
    elif graph == 'countplot':
      a = df.groupby([x,y]).count()
      a.reset_index(inplace = True)
      no_fraud = a[a['fraud_reported'] == 0]
      yes_fraud = a[a['fraud_reported'] == 1]
      trace1 = go.Bar(x = no_fraud[x], y = no_fraud['policy_number'], name = 'No Fraud')
      trace2 = go.Bar(x = yes_fraud[x], y = yes_fraud['policy_number'], name = 'Fraud')
      fig = go.Figure(data = [trace1, trace2])
      fig.update_layout(title = '{x} vs. {y}'.format(x=x, y = y))
      fig.update_layout(barmode = 'group')
      st.write(fig)


hobbies = df['insured_hobbies'].unique()
for hobby in hobbies:
  if (hobby != 'chess') & (hobby != 'cross-fit'):
    df['insured_hobbies'] = df['insured_hobbies'].str.replace(hobby, 'other')

#We will bin the ages and then check the trend for fraud vs. no fraud according to age.
df['age'].describe()
bin_labels = ['15-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61-65']
bins = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
df['age_group'] = pd.cut(df['age'], bins = bins, labels = bin_labels, include_lowest = True)

df['months_as_customer'].describe()
bin_labels = ['0-50', '51-100', '100-150', '151-200', '201-250', '251-300', '301-350', '351-400', '401-450', '451-500']
bins = [0,50,100,150,200,250,300,350,400,450,500]
df['month_group'] = pd.cut(df['months_as_customer'], bins = bins, labels = bin_labels, include_lowest = True)

df['policy_annual_premium'].describe()
bins = list(np.linspace(0,2500, 6, dtype = int))
bin_labels = ['very low', 'low', 'medium', 'high', 'very high']
df['policy_annual_premium_groups'] = pd.cut(df['policy_annual_premium'], bins = bins, labels=bin_labels)

df['policy_deductable'].describe()
bins = list(np.linspace(0,2000, 5, dtype = int))
bin_labels = ['0-500', '501-1000', '1001-1500', '1501-2000']
df['policy_deductable_group'] = pd.cut(df['policy_deductable'], bins = bins, labels = bin_labels)

required_columns1 = ['policy_number', 'insured_sex', 'insured_education_level', 'insured_occupation',
       'insured_hobbies', 'capital_gains', 'capital_loss', 'incident_type', 'collision_type', 'incident_severity',
       'authorities_contacted', 'incident_hour_of_the_day', 'number_of_vehicles_involved',
       'witnesses', 'total_claim_amount',
       'injury_claim', 'property_claim', 'vehicle_claim',
       'fraud_reported', 'age_group',
       'month_group', 'policy_annual_premium_groups']


df1 = df[required_columns1]

required_columns2 = ['insured_sex', 'insured_occupation',
       'insured_hobbies', 'capital_gains', 'capital_loss', 'incident_type', 'collision_type', 'incident_severity',
       'authorities_contacted', 'incident_hour_of_the_day', 'number_of_vehicles_involved',
       'witnesses', 'total_claim_amount', 'fraud_reported', 'age_group',
       'month_group', 'policy_annual_premium_groups']

df2 = df1[required_columns2]
num_features = df2._get_numeric_data().columns

df2['age_group'] = df2['age_group'].astype(object)
df2['month_group'] = df2['month_group'].astype(object)
df2['policy_annual_premium_groups'] = df2['policy_annual_premium_groups'].astype(object)

# extracting categorical columns
cat_df = df2.select_dtypes(include = ['object'])

# label endcoding for the object datatypes
from sklearn import preprocessing
for col in cat_df.columns:
    if (df2[col].dtype == 'object'):
        le = preprocessing.LabelEncoder()
        le = le.fit(df2[col])
        df2[col] = le.transform(df2[col])
        
#X = new_df[['months_as_customer', 'policy_csl', 'insured_sex','collision_type', 'incident_severity','authorities_contacted', 'incident_state', 'witnesses','injury_claim', 'property_claim','vehicle_claim', 'auto_make',]]
X_df = df2.drop(["fraud_reported"],axis=1)
y_df = df2['fraud_reported']

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=24)
X,y = sm.fit_resample(X_df, y_df)

#stadardize data    
from sklearn.preprocessing import StandardScaler
x_scaled = StandardScaler().fit_transform(X)

# splitting data into training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_scaled,y,test_size = 0.20)


pickle_in = open('rfc.pkl', 'rb')
rfc = pickle.load(pickle_in)


def prediction(umbrella_limit,incident_severity,bodily_injuries,witnesses,injury_claim,property_claim,vehicle_claim):
    prediction = rfc.predict([[umbrella_limit,incident_severity,bodily_injuries,witnesses,injury_claim,property_claim,vehicle_claim]])
    return prediction


# this is the main function in which we define our webpage
def main():
    html_temp = """
    <div style="background-color:#f63366 ;padding:10px;margin-bottom:10px;">
    <h1 style="color:white;text-align:center;">Fraud detection in Automobile Insurance Claims</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.sidebar.title("Pages")
    pages=['About Dataset','Exploratory Data Analysis', 'Data Preprocessing', 'Model Training' , 'Model Comparisons','Predictions']
    add_pages = st.sidebar.selectbox('', pages)

    if add_pages=='About Dataset':
        html_temp2 = """
        <body >
        <h3>About Dataset</h3>
        The objective of the dataset is to predict whether the insurance claims is valid or not using Machine learning algorithms.
        The datasets consists of several predictor variables and one target variable, Fraud Reported. 
        </body>
        """
        st.markdown(html_temp2, unsafe_allow_html=True)
        #Loading the dataset
        st.header("Preprocessed Dataset")
        st.subheader("First 5 Rows of the dataset")
        st.write(df.head())
        st.subheader("Last 5 Rows of the dataset")
        st.write(df.tail())
        st.write("The number of rows and columns:", df.shape)
        st.subheader("Dataset Description")
        st.write(df.describe().T)
        st.header("Target Variable")
        st.write('Fraud Report Cases: {}'.format(len(Fraud)))
        st.write('Valid Cases: {}'.format(len(Valid)))
        trace = go.Pie(labels = ['Valid','Fraud'], values = df['fraud_reported'].value_counts(), 
                   textfont=dict(size=15), opacity = 0.8,
                   marker=dict(colors=['green', 'red'], 
                               line=dict(color='#000000', width=1.5)))
        layout = dict(title =  'Distribution of Target variable')
        fig = dict(data = [trace], layout=layout)
        st.write(fig)
    if add_pages=='Exploratory Data Analysis':
        st.header("Data Distribution")
        #We will visualize the data and see if there is any feature which might influence the claims
        vis_data(df, 'insured_sex')
        vis_data(df, 'insured_education_level')
        vis_data(df, 'insured_occupation')
        vis_data(df, 'insured_relationship')
        vis_data(df, 'incident_type')
        vis_data(df, 'collision_type')
        vis_data(df, 'incident_severity')
        vis_data(df, 'authorities_contacted')
        vis_data(df, 'insured_hobbies')
        vis_data(df, 'age_group')
        vis_data(df, 'month_group')
        vis_data(df, 'auto_make')
        vis_data(df, 'number_of_vehicles_involved')
        vis_data(df, 'witnesses', 'fraud_reported')
        vis_data(df, 'bodily_injuries')
        vis_data(df, 'total_claim_amount', 'y', 'hist')
        vis_data(df, 'incident_hour_of_the_day')
        vis_data(df, 'number_of_vehicles_involved')
        vis_data(df, 'witnesses')
        vis_data(df, 'auto_year')
        vis_data(df, 'policy_annual_premium_groups')
        vis_data(df, 'policy_deductable_group')
        vis_data(df, 'police_report_available')

    if add_pages=='Data Preprocessing':
        st.header("Required columns")
        st.write("No.of Required Columns:",len(required_columns1))
        # checking for multicollinearity
        corr = df1.corr()
        fig = go.Figure(data = go.Heatmap( z = corr.values, x = list(corr.columns),y = list(corr.index),colorscale = 'Viridis'))
        fig.update_layout(title = 'Correlation')
        st.write(fig)
        st.write("From the correlation matrix, we see there is high correlation between vehicle claim, total_claim_amount, property_claim and injury_claim.")
        st.write("The reason for it is that total_claim_amount is the sum of columns vehicle claim,property_claim and injury_claim.")
        st.write("We will remove the other 3 columns and only keep total_claim_amount as it captures the information and removes collinearity.")
        st.write("No.of Required Columns:",len(required_columns2))
        st.write(df2.head())
        st.header("Extracting categorical columns")
        st.write(cat_df.head())
        st.header("Extracting Numerical columns")
        st.write(df2[num_features].head())
        st.header("Printing unique values of each column")
        for col in cat_df.columns:
            st.write(f"{col}: \n{cat_df[col].unique()}\n")
        st.header("Correlation")
        # Correlation matrix
        corrmat = df2.corr()
        fig = go.Figure(data = go.Heatmap( z = corrmat.values, x = list(corrmat.columns),y = list(corrmat.index),colorscale = 'Viridis'))
        fig.update_layout(title = 'Correlation')
        st.write()
        #Correlation with output variable
        cor_target = abs(corrmat["fraud_reported"])
        #Selecting highly correlated features
        relevant_features = cor_target[cor_target>0]
        st.write(relevant_features)

    if add_pages=="Model training":
        st.header("SMOTE(synthetic minority oversampling technique)")
        st.subheader("Actual Dataset")
        st.write(X_df.shape)
        st.write(y_df.shape)
        st.subheader("After Oversampling")
        st.write(X.shape)
        st.write(y.shape)
        st.header("Standard Scaler")
        st.write(x_scaled)
        st.header("Splitting Dataset")
        st.write(X_train.shape)
        st.write(X_test.shape)
        st.write(y_train.shape)
        st.write(y_test.shape)
        st.header("Logistic Regression")
        st.write("Train Set Accuracy:88.53")
        st.write("Test Set Accuracy:88.07" )
        st.header("Decision Tree Classifier")
        st.write("Train Set Accuracy:100")
        st.write("Test Set Accuracy:87.41" )
        st.header("Random Forest Classifier")
        st.write("Train Set Accuracy:100")
        st.write("Test Set Accuracy:90.06" )
        st.header("Support Vector Classifier")
        st.write("Train Set Accuracy:87.70")
        st.write("Test Set Accuracy:88.41" )
        st.header("Linear Discriminant Analysis")
        st.write("Train Set Accuracy:86.87")
        st.write("Test Set Accuracy:87.08" )

    if add_pages == 'Model Comparisons':
        st.header("Model Comparisons")
        models = pd.DataFrame({
         'Model': ['Logistic','Decision Tree Classifier','Random Forest Classifier','SVC','LDA'],
        'Score': [  0.880795,0.874172,0.900662,  0.884106,0.870861] })
        models.sort_values(by = 'Score', ascending = False)
        colors=['Logistic','Decision Tree Classifier','Random Forest Classifier','SVC','LDA']
        fig = px.bar(models, x='Model', y='Score',color=colors)
        st.write(fig)
    if add_pages=="Predictions":
        umbrella_limit = st.number_input("Umbrella Limit",min_value=0, max_value=6000000, value=0,step=1,format="%i")
        incident_severity=  st.number_input('Incident Severity',min_value=0, max_value=6000000, value=0,step=1,format="%i")
        bodily_injuries = st.number_input("Bodily Injuries:",min_value=0, max_value=6000000, value=0,step=1,format="%i")
        witnesses = st.number_input("Number of Witnesses:",min_value=0, max_value=6000000, value=0,step=1,format="%i")
        injury_claim = st.number_input("Injury Claim Amount",min_value=0, max_value=6000000, value=0,step=1,format="%i")
        property_claim =  st.number_input("Property Claim Amount",min_value=0, max_value=6000000, value=0,step=1,format="%i")
        vehicle_claim = st.number_input("Vehicle Claim Amount",min_value=0, max_value=6000000, value=0,step=1,format="%i")
        user_report_data = {'umbrella_limit':umbrella_limit,'incident_severity':incident_severity,'bodily_injuries':bodily_injuries,'witnesses':witnesses,'injury_claim':injury_claim,'property_claim':property_claim,'vehicle_claim':vehicle_claim}
        report_data = pd.DataFrame(user_report_data, index=[0])
        st.subheader('User Input Data')
        st.write(report_data)
        print(report_data.info())
        result =""
        st.subheader('Result: ')
        if st.button("Predict"):
            result = prediction(umbrella_limit,incident_severity,bodily_injuries,witnesses,injury_claim,property_claim,vehicle_claim)
            if result == 0:
                st.success('This is a Valid Automobile Insurance Claim')
            if result == 1:
                st.success('This is a Fraud Automobile Insurance Claim')


if __name__=='__main__':
    main()