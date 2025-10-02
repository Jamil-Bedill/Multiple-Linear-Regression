# Multiple Linear Regression with Penguins Data

## Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Key Steps in the Analysis](#key-steps-in-the-analysis)
  - [1. Importing Libraries & Data Loading](#1-importing-libraries--data-loading)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Feature Selection](#3-feature-selection)
  - [4. Model Construction (OLS Regression)](#4-model-construction-ols-regression)
  - [5. Checking Multicollinearity (VIF)](#5-checking-multicollinearity-vif)
- [Results](#results)

## Overview
This project applies **Multiple Linear Regression** on the Palmer Penguins dataset using Python.  
The goal is to predict penguin **body mass** based on **bill length**, **sex**, and **species**.  

This project demonstrates skills in **data cleaning, regression modelling, multicollinearity checks, and interpretation of coefficients**.

## Objectives
- Handle missing values and prepare dataset for regression.
- Construct a multiple linear regression model using `statsmodels`.
- Test assumptions including multicollinearity.
- Interpret regression coefficients in a business/biological context.

## Dataset
The dataset used is the **Palmer Penguins dataset** (Seaborn built-in).  
- **Shape**: 344 rows × 7 columns  
- **Features**:  
  - `species` – Penguin species (Adelie, Chinstrap, Gentoo)  
  - `island` – Island where penguins were observed  
  - `bill_length_mm` – Bill length in millimetres  
  - `bill_depth_mm` – Bill depth in millimetres  
  - `flipper_length_mm` – Flipper length in millimetres  
  - `body_mass_g` – Body mass in grams (target variable)  
  - `sex` – Male/Female  


## Key Steps in the Analysis  

### 1. Importing Libraries & Data Loading
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

```
Now loading the dataset and understanding it. 

```
# Load dataset
data = sns.load_dataset('penguins')
print(f'The dataset has {data.shape[0]} rows and  {data.shape[1]}')
print('------')
print('Null values\n')
print(data.isnull().sum())
print('------\n A few rows of dataset\n')
print(data.head())
```

<img width="663" height="499" alt="image" src="https://github.com/user-attachments/assets/daced36d-cf81-414d-b82a-b29ff555d26e" />

The dataset has some missing values. The next step would be to remove missing values. 

## 2. Data Cleaning

```
# Drop missing values
data.dropna(axis=0, inplace=True)

# Confirm
data.isnull().sum()
```
<img width="335" height="371" alt="image" src="https://github.com/user-attachments/assets/cd0cdfef-5e1c-46eb-ac4d-5739e3efafa7" />

## 3. Feature Selection
Here, I will use visualisation to understand variability in body mass due to penguin species and sex. 
First, I will plot a pairplot, which generally plots numerical variables

```
sns.pairplot(data)
```
<img width="556" height="525" alt="image" src="https://github.com/user-attachments/assets/26f4cf66-2398-4be8-a3b3-9e2a3fed2a53" />

We can understand that there is a  linear relationship between body mass and bill length. We include this in the model. 
Now I will use boxplots to understand the relationship between body mass and sex and body mass and species because the two variables, species and sex, are categorical variables. 

```
fig, axes = plt.subplots(1,2, figsize = (8,4))
sns.boxplot(x = 'sex', y = 'body_mass_g' , data= data, ax = axes[0])
axes[0].set_xlabel('Sex of Penguins')
axes[0].set_ylabel('Body mass of Penguins')
axes[0].set_title('Box Plot of Penguins Sex vs body mass')
sns.boxplot(x = 'species', y = 'body_mass_g' , data= data , ax = axes[1])
axes[1].set_xlabel('Species of Penguins')
axes[1].set_ylabel('Body mass of Penguins')
axes[1].set_title('Box Plot of Penguins Species vs body mass')
plt.tight_layout()
plt.show()
```
<img width="864" height="396" alt="image" src="https://github.com/user-attachments/assets/84c826f3-faa4-4019-a8fa-91fcef6e0421" />

Now there is  a higher variability in body mass based on sex and species. So we include these two variables as well in the model. 

```
# Now let's get the necessary columns from the dataset
data = data[['body_mass_g', 'bill_length_mm', 'sex', 'species']]
data.reset_index(inplace = True, drop = True)
# confirming
data.head()
```
<img width="415" height="193" alt="image" src="https://github.com/user-attachments/assets/3548399a-7ecd-4b44-9340-8c2bb93adc8e" />

We can now split dependent variablel(body_mass_g) and independent variables(species, sex, bill_length_mm)

```
# Split features and target
data_x = data[['bill_length_mm', 'sex', 'species']]
data_y = data['body_mass_g']
```
## 4. Model Construction (OLS Regression)
In this stage, We import the necessary libraries, split the data into train and test, fit the model, and show the summary. 

```
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=42)

# Prepare data for OLS
ols_data = pd.concat([x_train, y_train], axis=1)

# Build regression model
ols_formula = 'body_mass_g ~ bill_length_mm + C(sex) + C(species)'
OLS = ols(formula=ols_formula, data=ols_data)
model = OLS.fit()

print(model.summary())
```
<img width="821" height="499" alt="image" src="https://github.com/user-attachments/assets/b1a71484-7583-489d-8b90-2613309c19cf" />

We fitted an Ordinary Least Squares (OLS) regression model to predict penguin body mass (grams) using species, sex, and bill length as predictors.

**Model Fit**: The model explains 85% of the variation in body mass (R² = 0.850, Adjusted R² = 0.847).

**Statistical Significance**: The model is highly significant (F = 322.6, p < 0.001).

#### Key Findings:

**Sex effect**: Male penguins are on average 529 g heavier than females (p < 0.001), controlling for species and bill length.

**Species effect**: Chinstrap penguins are about 285 g lighter than Adelie penguins (p = 0.008), holding sex and bill length constant. Gentoo penguins are about 1082 g heavier than Adelie penguins (p < 0.001), holding sex and bill length constant.

**Bill length**: Each additional millimetre in bill length is associated with a 35.6 g increase in body mass (p < 0.001), controlling for sex and species.

**Assumption Checks**: Residuals appear normally distributed (Omnibus p = 0.844; JB p = 0.804) and independent (Durbin-Watson = 1.95).
Multiple linear regression has the same assumptions as simple linear regression, but has one more assumption: no multicollinearity. For other assumptions, we can use the same method. Please check my repository for simple linear regression. We will only check the no multicollinearity assumption. 
## 5. Checking Multicollinearity (VIF)
We first need to change categorical variables into numeric variables and then check the variance Inflation factor(VIF)
**Thresholds for VIF**
- VIF < 5 (or VIF < 10): This is the acceptable range. Multicollinearity is considered low enough not to impact the reliability or stability of the model's coefficients.
- VIF > 5 (or VIF > 10): This suggests strong correlation, and you would usually investigate dropping the variable with the highest score.

```
# Let's test multicolinearity assumption using vif method
penguin_dummies = pd.get_dummies(data[['bill_length_mm', 'sex', 'species']], drop_first= True)
penguin_dummies = penguin_dummies.dropna(axis=0)
penguin_dummies = penguin_dummies.astype(float)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data['Variable']= penguin_dummies.columns
vif_data['VIF'] = [variance_inflation_factor(penguin_dummies.values, i) for i in range(penguin_dummies.shape[1])]
print(vif_data)
```
<img width="287" height="99" alt="image" src="https://github.com/user-attachments/assets/7ba683db-694c-40b6-a424-3011b5be44c7" />

 **Analysis of the Score**
**Highest Score (bill_length_mm at 4.66)**: While this is the highest score, it is still below the threshold of 5. This score is expected because a penguin's bill length is highly determined by its sex and species. The VIF tells you that about 78.5% of the variance in bill_length_mm is explained by the combination of the species and sex dummy variables. This is not a problem for the model; it just confirms the variables are related (which is true in nature).

**Dummy Variables (sex and species**): All dummy variables have very low VIF scores (below 2.3). 
All VIF < 5 → No problematic multicollinearity.

## Results

- Body mass is strongly influenced by bill length, sex, and species.
- Male penguins are significantly heavier than females.
- Gentoo penguins are the heaviest, while Chinstrap penguins are slightly lighter compared to Adelie.
- Model explains 85% of the variation in penguin body mass, making it a robust predictive tool.
