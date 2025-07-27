#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
df = pd.read_csv('../dataset/student-social-media-addiction.csv')
#%%

print(df.head())

print(df.isna().sum())

#%%
df_clean = df.dropna()
print(df.describe())


# %%
print(df.columns.to_list())
print(df.dtypes)

# %%
print(df[(df.iloc[:,1] > 20) & (df.iloc[:,5] > 2)].iloc[:,[1,5]].head())

# %%
print(df['Country'].value_counts())


# %%
print(df.groupby('Academic_Level')['Avg_Daily_Usage_Hours'].mean())

# %%
print((df['Sleep_Hours_Per_Night']<6).sum())

# %%
print(df[['Student_ID','Addicted_Score']].sort_values(by='Addicted_Score', ascending=False).head(5))

# %%
print(df.groupby('Relationship_Status')['Mental_Health_Score'].mean())

# %%
df['Avg_Daily_Usage_Hours'].plot(kind='hist', bins=10)
plt.xlabel('Avg Daily Usage Hours')
plt.ylabel('Count of Students')
plt.title('Distribution of Average Daily Usage Hours')
plt.show()

# %%
df.groupby('Relationship_Status')['Mental_Health_Score'].mean().plot(kind='bar')
plt.ylabel('Average Mental Health Score')
plt.title('Mental Health Score by Academic Level')
plt.show()




# %%
