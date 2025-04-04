

# Sentiment Analysis of Amazon Reviews Using NLP and Data Visualization
# Introduction
Customer reviews are crucial for understanding consumer sentiment and product performance. This analysis explores Amazon product reviews to classify sentiment using Natural Language Processing (NLP) and visualizations. The dataset includes customer feedback, star ratings, and review text, allowing us to analyze trends in sentiment and review patterns. Using VADER sentiment analysis, we classified reviews into positive, negative, and neutral sentiments. Additionally, creative visualizations such as word clouds, violin plots, scatter plots, and heatmaps provided deeper insights into customer opinions.

# Data Exploration and Processing
We loaded the dataset and selected 500 reviews for analysis. The Score column (ranging from 1 to 5 stars) represented product ratings, while the Text column contained the actual review. We applied tokenization, Part-of-Speech (POS) tagging, and Named Entity Recognition (NER) to process textual data. Using VADER, we assigned a sentiment score to each review, classifying them as positive (>0.05), negative (<-0.05), or neutral (-0.05 to 0.05).

# Review Score Distribution
A histogram with KDE (Kernel Density Estimation) revealed that most reviews were positive (4-5 stars), while negative ratings (1-2 stars) were fewer. This trend aligns with typical consumer behavior, where satisfied customers leave high ratings. We also created a scatter plot to examine how sentiment scores varied across review ratings.

# Word Cloud Analysis
A word cloud visualized the most commonly used words in reviews. Positive reviews frequently contained words like great, love, and excellent, whereas negative reviews included words like bad, disappointed, and poor. This analysis provided a quick summary of common customer sentiments.

# Sentiment Analysis and Visualization
We used violin plots to show the distribution of sentiment scores across star ratings, revealing that higher ratings strongly correlate with positive sentiment. A swarm plot further visualized the spread of sentiment across different review scores. Additionally, a heatmap highlighted the correlation between Score and sentiment scores, confirming that higher ratings align with positive sentiment.

# Conclusion
This sentiment analysis effectively demonstrated how NLP and data visualization can provide meaningful insights into customer reviews. The VADER sentiment analysis, word cloud, and advanced visualizations helped uncover patterns in customer feedback. These insights can help businesses improve products, enhance customer satisfaction, and refine marketing strategies. Future work could involve deep learning-based sentiment models for more accurate classification.
