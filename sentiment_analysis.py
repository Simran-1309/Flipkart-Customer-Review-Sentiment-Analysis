from wordcloud import WordCloud, STOPWORDS
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

# Load cleaned dataset
df = pd.read_csv("Dataset/cleaned_reviews.csv")

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Calculate sentiment score
df["vader_score"] = df["Review"].apply(
    lambda review: sia.polarity_scores(str(review))["compound"]
)

# Classify sentiment
def classify_sentiment(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df["vader_sentiment"] = df["vader_score"].apply(
    classify_sentiment
)

# Display results
print("\nVADER Sentiment Analysis Completed!")

print("\nSentiment Distribution:")
print(df["vader_sentiment"].value_counts())

print("\nSample Results:")
print(df[["Review", "vader_score", "vader_sentiment"]].head(10))

# Save results
df.to_csv(
    "Dataset/vader_analyzed_reviews.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nAnalyzed dataset saved successfully!")


import matplotlib.pyplot as plt
import seaborn as sns

# Count sentiments
sentiment_counts = df["vader_sentiment"].value_counts()

# Create bar chart
plt.figure(figsize=(8, 5))

sns.barplot(
    x=sentiment_counts.index,
    y=sentiment_counts.values
)

plt.title("Flipkart Customer Review Sentiment Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.tight_layout()

# Save chart
plt.savefig(
    "Visualizations/sentiment_distribution.png"
)

plt.show()

print("Sentiment chart saved successfully!")
# Check customer rating distribution


print("\nRating Data Type:")
print(df["Rate"].dtype)
# Clean and convert ratings to numeric values
df["Rate"] = pd.to_numeric(
    df["Rate"].astype(str).str.strip(),
    errors="coerce"
)

print("\nRating Data Type:")
print(df["Rate"].dtype)

print("\nInvalid or Missing Ratings:")
print(df["Rate"].isna().sum())

print("\nCustomer Rating Distribution:")
print(df["Rate"].value_counts(dropna=False).sort_index())

print("\nRating vs VADER Sentiment:")

rating_sentiment = pd.crosstab(
    df["Rate"],
    df["vader_sentiment"]
)

print(rating_sentiment)
# Calculate sentiment percentages for each rating
rating_sentiment_percent = pd.crosstab(
    df["Rate"],
    df["vader_sentiment"],
    normalize="index"
) * 100

# Create heatmap
plt.figure(figsize=(8, 5))

sns.heatmap(
    rating_sentiment_percent,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    cbar_kws={"label": "Percentage"}
)

plt.title("Customer Rating vs VADER Sentiment")
plt.xlabel("VADER Sentiment")
plt.ylabel("Customer Star Rating")

plt.tight_layout()

# Save heatmap
plt.savefig(
    "Visualizations/rating_vs_sentiment_heatmap.png"
)

plt.show()

print("Rating vs Sentiment heatmap saved successfully!")
# Product-level sentiment analysis
product_sentiment = df.groupby("product_name").agg(
    Total_Reviews=("Review", "count"),
    Positive_Reviews=(
        "vader_sentiment",
        lambda x: (x == "positive").sum()
    ),
    Negative_Reviews=(
        "vader_sentiment",
        lambda x: (x == "negative").sum()
    )
).reset_index()

product_sentiment["Positive_Percentage"] = (
    product_sentiment["Positive_Reviews"]
    / product_sentiment["Total_Reviews"] * 100
).round(2)

print("\nTop 10 Products by Review Count:")
print(
    product_sentiment.sort_values(
        by="Total_Reviews",
        ascending=False
    ).head(10)
)

# Save product analysis
product_sentiment.to_csv(
    "Dataset/product_sentiment_analysis.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nProduct sentiment analysis saved successfully!")

print("\nTop 10 Products by Positive Sentiment:")

print(
    product_sentiment[
        product_sentiment["Total_Reviews"] >= 20
    ]
    .sort_values(
        by="Positive_Percentage",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)




# Top 10 products by positive sentiment
top_products = (
    product_sentiment[
        product_sentiment["Total_Reviews"] >= 20
    ]
    .sort_values(
        by="Positive_Percentage",
        ascending=False
    )
    .head(10)
)
top_products["product_name"] = (
    top_products["product_name"]
    .str.encode("ascii", errors="ignore")
    .str.decode("ascii")
)
plt.figure(figsize=(12, 7))

sns.barplot(
    data=top_products,
    x="Positive_Percentage",
    y="product_name"
)

plt.title("Top 10 Products by Positive Sentiment")
plt.xlabel("Positive Sentiment (%)")
plt.ylabel("Product Name")

plt.tight_layout()

plt.savefig(
    "Visualizations/top_positive_products.png"
)

plt.show()

print("Top positive products chart saved successfully!")

# Top 10 products by negative sentiment
top_negative_products = (
    product_sentiment[
        product_sentiment["Total_Reviews"] >= 20
    ]
    .sort_values(
        by="Negative_Reviews",
        ascending=False
    )
    .head(10)
)

print("\nTop 10 Products by Negative Reviews:")
print(top_negative_products.to_string(index=False))

# Calculate negative sentiment percentage
product_sentiment["Negative_Percentage"] = (
    product_sentiment["Negative_Reviews"]
    / product_sentiment["Total_Reviews"] * 100
).round(2)

print("\nTop 10 Products by Negative Sentiment Percentage:")

print(
    product_sentiment[
        product_sentiment["Total_Reviews"] >= 20
    ]
    .sort_values(
        by="Negative_Percentage",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


top_negative_percentage = (
    product_sentiment[
        product_sentiment["Total_Reviews"] >= 20
    ]
    .sort_values(
        by="Negative_Percentage",
        ascending=False
    )
    .head(10)
)

top_negative_percentage["product_name"] = (
    top_negative_percentage["product_name"]
    .str.encode("ascii", errors="ignore")
    .str.decode("ascii")
)

plt.figure(figsize=(12, 7))

sns.barplot(
    data=top_negative_percentage,
    x="Negative_Percentage",
    y="product_name"
)

plt.title("Top 10 Products by Negative Sentiment")
plt.xlabel("Negative Sentiment (%)")
plt.ylabel("Product Name")

plt.tight_layout()

plt.savefig(
    "Visualizations/top_negative_products.png"
)

plt.show()

print("Top negative products chart saved successfully!")


# Generate Positive and Negative Word Clouds

positive_text = " ".join(
    df[df["vader_sentiment"] == "positive"]["Review"].astype(str)
)

negative_text = " ".join(
    df[df["vader_sentiment"] == "negative"]["Review"].astype(str)
)

# Positive Word Cloud
positive_wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white",
    stopwords=STOPWORDS,
    collocations=False
).generate(positive_text)

plt.figure(figsize=(12, 6))
plt.imshow(positive_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Customer Review Word Cloud")
plt.savefig("Visualizations/positive_wordcloud.png")
plt.show()

# Negative Word Cloud
negative_wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white",
    stopwords=STOPWORDS,
    collocations=False
).generate(negative_text)

plt.figure(figsize=(12, 6))
plt.imshow(negative_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Customer Review Word Cloud")
plt.savefig("Visualizations/negative_wordcloud.png")
plt.show()

print("Both Word Clouds saved successfully!")


# Sentiment Percentage by Customer Rating

rating_analysis = pd.crosstab(
    df["Rate"],
    df["vader_sentiment"],
    normalize="index"
) * 100

rating_analysis = rating_analysis.round(2)

print("\nSentiment Percentage by Customer Rating:")
print(rating_analysis)


# Rating-wise Sentiment Percentage Chart

rating_analysis.plot(
    kind="bar",
    figsize=(10, 6),
    width=0.8
)

plt.title("Sentiment Percentage by Customer Rating")
plt.xlabel("Customer Star Rating")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=0)
plt.legend(title="VADER Sentiment")
plt.tight_layout()

plt.savefig(
    "Visualizations/rating_wise_sentiment.png"
)

plt.show()

print("Rating-wise sentiment chart saved successfully!")