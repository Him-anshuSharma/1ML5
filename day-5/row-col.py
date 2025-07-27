import pandas as pd

#creating data frame (2d table)
df = pd.read_csv('../dataset/Titanic-Dataset.csv')

#first two rows and column age and name
print("\nFirst two rows and columns 'Age' and 'Name:'\n",df.loc[0:1,['Age','Name']])

#first two rows and columns from indices
print("\nFirst two rows and columns from indices 2 & 3:\n",df.iloc[0:1,2:4])

#filtering rows based on a condition
filtered =  df[(df['Age'] > 30) & (df['Survived'] == 1)]
names = filtered[['Name']]
print("Names where Age is greater than 30:\n", names)

#exercise
filtered = df.loc[(df['Age'] > 30) | (df['Survived'] == 1), ['Name','Age','Sex']]
print("\nFiltered names where Age is greater than 30 or Survived is 1:\n", filtered)