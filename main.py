import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('credit.csv')

## CLEANING DATA

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

## END CLEANING DATA

## Neural Network

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Pre-elaborazione dei dati
preprocessor = make_column_transformer(
    (StandardScaler(), ['age', 'debt', 'yearemployed', 'creditScore', 'income']),
    (OneHotEncoder(), ['Gender', 'married', 'BankCustomer', 'Ethnicity', 'PriorDefault', 'Employed', 'DriversLicense', 'Citizen'])
)

X = df.drop('Approved', axis=1)
X = preprocessor.fit_transform(X)
y = df['Approved']

# Divisione del dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Creazione del modello di rete neurale
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilazione del modello
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Addestramento del modello
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Valutazione del modello
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')











