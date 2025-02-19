import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from wordcloud import WordCloud


plt.style.use('seaborn-v0_8-darkgrid')  



df = pd.read_csv(r"C:\Users\joyfu.DESKTOP-5ETBPHM\OneDrive\Desktop\puneeth\codtech internship\task 4\Reviews.csv")
df = df.head(500) 

sns.histplot(df['Score'], bins=5, kde=True, color="purple")
plt.title("📊 Distribution of Review Scores", fontsize=14, fontweight="bold")
plt.xlabel("Review Stars")
plt.ylabel("Frequency")
plt.show()


text = " ".join(review for review in df["Text"].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("🌟 Most Frequent Words in Reviews", fontsize=14, fontweight="bold")
plt.show()

# ----------------------- Sentiment Analysis -----------------------
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Apply sentiment analysis
df["sentiment"] = df["Text"].astype(str).apply(lambda x: sia.polarity_scores(x)["compound"])
df["sentiment_category"] = df["sentiment"].apply(lambda x: "Positive" if x > 0.05 else ("Negative" if x < -0.05 else "Neutral"))

# ----------------------- Boxplot of Sentiment Scores by Review Stars -----------------------
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="Score", y="sentiment", palette="coolwarm")
plt.title("📉 Sentiment Scores by Review Stars", fontsize=14, fontweight="bold")
plt.xlabel("Review Stars")
plt.ylabel("Sentiment Score")
plt.show()

# ----------------------- Violin Plot for Sentiment Score Distribution -----------------------
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x="Score", y="sentiment", palette="plasma")
plt.title("🎻 Sentiment Score Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Review Stars")
plt.ylabel("Sentiment Score")
plt.show()

# ----------------------- Swarm Plot for Sentiment Category -----------------------
plt.figure(figsize=(12, 6))
sns.swarmplot(data=df, x="Score", y="sentiment", hue="sentiment_category", palette={"Positive": "green", "Neutral": "blue", "Negative": "red"})
plt.title("🦟 Sentiment Category by Review Score", fontsize=14, fontweight="bold")
plt.xlabel("Review Stars")
plt.ylabel("Sentiment Score")
plt.legend(title="Sentiment Category")
plt.show()

# ----------------------- Heatmap of Correlations -----------------------
plt.figure(figsize=(10, 6))
sns.heatmap(df[["Score", "sentiment"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("🔥 Correlation Heatmap", fontsize=14, fontweight="bold")
plt.show()

# ----------------------- Scatter Plot for Sentiment vs Review Score -----------------------
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x="Score", y="sentiment", hue="sentiment_category", palette={"Positive": "green", "Neutral": "blue", "Negative": "red"}, alpha=0.7)
plt.title("🎯 Sentiment Score vs Review Stars", fontsize=14, fontweight="bold")
plt.xlabel("Review Stars")
plt.ylabel("Sentiment Score")
plt.legend(title="Sentiment Category")
plt.show()

# ----------------------- Density Plot for Sentiment Scores -----------------------
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df, x="sentiment", hue="sentiment_category", fill=True, palette={"Positive": "green", "Neutral": "blue", "Negative": "red"}, alpha=0.5)
plt.title("🌊 Sentiment Score Density Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Sentiment Score")
plt.show()
