import pandas as pd
import statsmodels.api as sm

# Load the dataset
data = pd.read_csv('data1.csv')

# Preprocess the data
data['Ovulation Problem'] = data['Ovulation Problem'].map({'yes': 1, 'no': 0})
data['Fertility Problem'] = data['Fertility Problem'].map({'yes': 1, 'no': 0})
data['Pregnant Before'] = data['Pregnant Before'].map({'yes': 1, 'no': 0})
data['Tubes Issue'] = data['Tubes Issue'].map({'yes': 1, 'no': 0})
data['Male Fertility Issue'] = data['Male Fertility Issue'].map({'yes': 1, 'no': 0})

# Split the data into features and target variable
X = data[['age', 'Years Trying', 'Ovulation Problem', 'Fertility Problem', 'Pregnant Before', 'Tubes Issue', 'Male Fertility Issue']]
y = data['Success Rate']

# Add a constant column to the features
X = sm.add_constant(X)

# Train the OLS model
model = sm.OLS(y, X)
results = model.fit()

# Print the model summary
print(results.summary())
