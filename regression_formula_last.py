import pandas as pd
import statsmodels.api as sm

# Load the dataset
data = pd.read_csv('data1.csv')

# Preprocess the data
data['Ovulation Problem'] = data['Ovulation Problem'].map({'yes': 1, 'no': 0})
data['Fertility Problem'] = data['Fertility Problem'].map({'yes': 1, 'no': 0})
data['Pregnant Before'] = data['Pregnant Before'].map({'yes': 1, 'no': 0})
data['Tubes issue'] = data['Tubes issue'].map({'yes': 1, 'no': 0})
data['Male fertility issue'] = data['Male fertility issue'].map({'yes': 1, 'no': 0})

# Split the data into features and target variable
X = data[['Age', 'Years Trying', 'Ovulation Problem', 'Fertility Problem', 'Pregnant Before', 'Tubes issue', 'Male fertility issue']]
y = data['Success Rate']

# Add a constant column to the features
X = sm.add_constant(X)

# Train the OLS model
model = sm.OLS(y, X)
results = model.fit()

# Print the model summary
print(results.summary())

# Extract the coefficients and variable names
coefs = results.params[1:]
variables = X.columns[1:]

# Create the formula
formula = 'Success Rate = ' + str(results.params[0])

for coef, variable in zip(coefs, variables):
    formula += ' + ' + str(coef) + ' * ' + variable

# Print the formula
print('\nFormula: ', formula)




# could you get me the formula from this OLS Regression Results                            
# ==============================================================================
# Dep. Variable:           Success Rate   R-squared:                       0.726
# Model:                            OLS   Adj. R-squared:                  0.692
# Method:                 Least Squares   F-statistic:                     21.59
# Date:                Mon, 05 Jun 2023   Prob (F-statistic):           6.57e-14
# Time:                        18:13:53   Log-Likelihood:                -224.88
# No. Observations:                  65   AIC:                             465.8
# Df Residuals:                      57   BIC:                             483.1
# Df Model:                           7                                         
# Covariance Type:            nonrobust                                         
# ========================================================================================
#                            coef    std err          t      P>|t|      [0.025      0.975]
# ----------------------------------------------------------------------------------------
# const                   68.0715      3.946     17.251      0.000      60.170      75.973
# Age                     -1.1464      0.116     -9.901      0.000      -1.378      -0.915
# Years Trying            -0.3801      0.183     -2.072      0.043      -0.747      -0.013
# Ovulation Problem       -0.8575      3.011     -0.285      0.777      -6.886       5.171
# Fertility Problem       -0.4590      3.217     -0.143      0.887      -6.900       5.982
# Pregnant Before         -4.4157      3.297     -1.339      0.186     -11.017       2.186
# Tubes issue              1.5653      3.121      0.502      0.618      -4.684       7.815
# Male fertility issue    -4.7674      3.712     -1.284      0.204     -12.201       2.666
# ==============================================================================
# Omnibus:                       13.011   Durbin-Watson:                   0.428
# Prob(Omnibus):                  0.001   Jarque-Bera (JB):                3.482
# Skew:                           0.065   Prob(JB):                        0.175
# Kurtosis:                       1.874   Cond. No.                         148.
# ==============================================================================

# ChatGPT
# Certainly! The formula for the OLS regression model you provided is:

# Success Rate = 68.0715 - 1.1464 * Age - 0.3801 * Years Trying - 0.8575 * Ovulation Problem - 0.4590 * Fertility Problem - 4.4157 * Pregnant Before + 1.5653 * Tubes issue - 4.7674 * Male fertility issue

# In this formula, each coefficient represents the estimated effect of the corresponding independent variable on the dependent variable (Success Rate). The "const" term represents the intercept or the baseline value of Success Rate when all independent variables are zero. The t-values and p-values associated with each coefficient indicate the statistical significance of the estimated effects.



# To estimate the Success Rate using the given data and the formula:

# Success Rate ≈ Base Rate + A * Age + B * Years Trying + C * Ovulation Problem + D * Fertility Problem + E * Pregnant Before + F * Tubes issue + G * Male fertility issue

# We need to perform a regression analysis to determine the values of the constants (A, B, C, D, E, F, G) that would best fit the data. Regression analysis is a statistical technique used to model the relationship between a dependent variable (Success Rate in this case) and one or more independent variables (Age, Years Trying, Ovulation Problem, Fertility Problem, Pregnant Before, Tubes issue, Male fertility issue).

# The given data consists of comma-separated values representing different factors and their corresponding Success Rates. To perform the regression analysis, we first need to separate the data into columns and convert them into numerical values.

# Here is the separated data:

# Age	Years Trying	Ovulation Problem	Fertility Problem	Pregnant Before	Tubes issue	Male fertility issue	Success Rate
# 18	0	no	no	no	no	no	39.21%
# 18	0	yes	no	no	no	no	40.39%
# ...	...	...	...	...	...	...	...
# 50	20	no	no	yes	no	no	0.4%
# To perform the regression analysis and estimate the values of the constants, you can use statistical software or programming languages that provide regression analysis capabilities such as Python with libraries like scikit-learn or R with packages like stats. The regression analysis will help you find the best-fit values for the constants that minimize the difference between the estimated Success Rate and the actual Success Rate in the data.

# Keep in mind that the formula you provided is a linear model, assuming a linear relationship between the independent variables and the Success Rate. If you suspect non-linear relationships, you may need to explore other regression techniques such as polynomial regression or consider using more complex models like neural networks.



# User
# could you give me the polynomial regression and give me the right formula Success Rate ≈ Base Rate + A * Age + B * Years Trying + C * Ovulation Problem + D * Fertility Problem + E * Pregnant Before + F * Tubes issue + G * Male fertility issue
# ChatGPT
# Certainly! To perform polynomial regression and find the best-fit formula for estimating the Success Rate, you can follow these steps:

# Step 1: Prepare the Data

# Separate the given data into columns: Age, Years Trying, Ovulation Problem, Fertility Problem, Pregnant Before, Tubes issue, Male fertility issue, and Success Rate.
# Convert any categorical variables (such as "no" and "yes") into numerical values if needed.
# Step 2: Fit the Polynomial Regression Model

# Use a regression library or module in a programming language like Python (e.g., scikit-learn) or R (e.g., stats) to fit the polynomial regression model to the data.
# Specify the degree of the polynomial you want to use. For example, if you choose degree 2, the model will include quadratic terms.
# Fit the model using the independent variables (Age, Years Trying, Ovulation Problem, Fertility Problem, Pregnant Before, Tubes issue, Male fertility issue) and the dependent variable (Success Rate).
# Step 3: Analyze the Model and Obtain Coefficients

# Once the model is fitted, you can analyze the coefficients of the polynomial regression model to determine their significance and impact on the Success Rate.
# The coefficients (A, B, C, D, E, F, G) represent the impact of each independent variable on the Success Rate, considering the polynomial terms if applicable.
# Step 4: Formulate the Final Formula

# Based on the coefficients obtained from the polynomial regression model, you can formulate the final formula for estimating the Success Rate.
# The formula will be in the form: Success Rate ≈ Base Rate + A * Age + B * Years Trying + C * Ovulation Problem + D * Fertility Problem + E * Pregnant Before + F * Tubes issue + G * Male fertility issue, with the appropriate coefficient values substituted.
# Please note that I can provide guidance and general steps, but implementing polynomial regression and obtaining the specific coefficients would require programming and data analysis skills.