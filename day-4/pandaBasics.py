import pandas as pd

#creating data frame (2d table)
df = pd.read_csv('../dataset/Titanic-Dataset.csv')

#head: loads first 5 rows of the data frame
print("\nHead:\n", df.head(),"\n")

#info: information about the data frame
print("\nInfo:\n", df.info(),"\n")

#describe: statistical summary of the data frame
print("\nDescribe:\n", df.describe(),"\n")

#rows and columns
print("\nShape:", df.shape[0], "rows and", df.shape[1], "columns\n")

#dtypes: data types of each column
print("\nData of column types:\n", df.dtypes,"\n")

#isnull: checks for null values in the data frame
print("\nNull values:\n", df.isnull().sum(),"\n")
