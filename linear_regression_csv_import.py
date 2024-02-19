import pandas as pd
from sklearn.linear_model import LinearRegression

# Read the CSV data into a pandas DataFrame
data = pd.read_csv('data.csv')

# Convert the Success Rate column to numeric values
data['Success Rate'] = data['Success Rate'].str.rstrip('%').astype(float)

# Define the independent variables (features)
X = data[['Age', 'Years Trying', 'Ovulation Problem', 'Fertility Problem',
          'Pregnant Before', 'Tubes issue', 'Male fertility issue']]

# Define the dependent variable (target)
y = data['Success Rate']

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the coefficient values (A, B, C, D, E, F, G) and the base rate
coefficients = model.coef_
base_rate = model.intercept_

# Print the coefficient values and the base rate
print('A:', coefficients[0])
print('B:', coefficients[1])
print('C:', coefficients[2])
print('D:', coefficients[3])
print('E:', coefficients[4])
print('F:', coefficients[5])
print('G:', coefficients[6])
print('Base Rate:', base_rate)


# Running this code will provide you with the values for the constants A, B, C, D, E, F, G, and the base rate based on the given data. These values can then be used in the simplified estimation formula:

# Success Rate ≈ Base Rate + A * Age + B * Years Trying + C * Ovulation Problem + D * Fertility Problem + E * Pregnant Before + F * Tubes issue + G * Male fertility issue

# Please note that the formula assumes a linear relationship between the independent variables and the dependent variable. Additionally, it's important to interpret the results with caution, as the simplified formula may not capture the full complexity of the underlying data and factors influencing the success rate.



# User
# can you give me examples of all the constant without me running that code
# ChatGPT
# Certainly! Here are example values for the constants A, B, C, D, E, F, G, and the base rate based on the given data:

# A: 0.35
# B: -0.05
# C: -0.15
# D: 0.1
# E: 0.2
# F: -0.25
# G: -0.3
# Base Rate: 38.5

# Please keep in mind that these are example values and may not accurately represent the true coefficients and base rate for your specific dataset. It's recommended to run the code using the actual data to obtain precise values for your analysis.