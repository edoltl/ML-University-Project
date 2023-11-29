import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('credit.csv')

# describe columns type
print(df.dtypes)

# print first 5 rows
print(df.head())

## CLEANING DATA

# clean data and show how many rows are dropped
print(df.shape)
df = df.dropna()
print(df.shape)

# Change X to ID
df = df.rename(columns={'X': 'ID'})

# Change column name male to Gender
df = df.rename(columns={'male': 'Gender'})

# Change gender to 0 and 1
df['Gender'] = df['Gender'].map({'a': 0, 'b': 1})

# Change married to 0 and 1
df['married'] = df['married'].map({'u': 1, 'y': 0, 'l': 0})

# Change backCustomer to BankCustomer and change to 0 and 1
df = df.rename(columns={'backCustomer': 'BankCustomer'})
df['BankCustomer'] = df['BankCustomer'].map({'g': 1, 'gg': 0, 'p': 0})

# change etnicity to White, Black, Latin, Other
df = df.rename(columns={'etnicity': 'Ethnicity'})
df['Ethnicity'] = df['Ethnicity'].map({'v': "White", 'h': "Black", 'z': "Other", 'o': "Other", 'n': "Other", 'ff': "Latin", 'j': "Other", 'dd': "Other", 'bb': "Other"})

# change PriorDefault to 0 and 1
df = df.rename(columns={'priordefault': 'PriorDefault'})
df['PriorDefault'] = df['PriorDefault'].map({'t': 1, 'f': 0})

# change Employed to 0 and 1
df = df.rename(columns={'employed': 'Employed'})
df['Employed'] = df['Employed'].map({'t': 1, 'f': 0})

# change DriversLicense to 0 and 1
df = df.rename(columns={'driverlicence': 'DriversLicense'})
df['DriversLicense'] = df['DriversLicense'].map({'t': 1, 'f': 0})

# change Citizen to byBirth, byOtherMeans, Temporary
df = df.rename(columns={'citizen': 'Citizen'})
df['Citizen'] = df['Citizen'].map({'g': "byBirth", 's': "byOtherMeans", 'p': "Temporary"})

# change approved to 0 and 1
df = df.rename(columns={'approved': 'Approved'})
df['Approved'] = df['Approved'].map({'-': 0, '+': 1})

print(df.head())

## END CLEANING DATA









