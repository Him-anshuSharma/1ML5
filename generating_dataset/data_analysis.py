import pandas as pd

df = pd.read_csv('subtle_emotion_journals.csv')

emotion_cols = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Summary statistics
print(df[emotion_cols].describe())

# Visualize distribution
import matplotlib.pyplot as plt

df[emotion_cols].sum().plot(kind="bar", title="Number of examples per emotion")
plt.show()

df["text_length"] = df["journal"].apply(len)
print(df["text_length"].describe())

# Histogram
df["text_length"].hist(bins=30)
plt.title("Distribution of journal lengths")
plt.show()

# Count entries with multiple strong emotions
df["num_strong_emotions"] = (df[emotion_cols] > 0.5).sum(axis=1)
print(df["num_strong_emotions"].value_counts())
