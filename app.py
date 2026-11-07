
import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Flipkart Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Sidebar
st.sidebar.title("📊 Flipkart Analytics")

st.sidebar.write("### Project Information")
st.sidebar.write("**Project:** Customer Review Sentiment Analysis")
st.sidebar.write("**Technique:** VADER Sentiment Analysis")
st.sidebar.write("**Dataset:** Flipkart Customer Reviews")

st.sidebar.divider()

st.sidebar.write("### Technologies Used")
st.sidebar.write("- Python")
st.sidebar.write("- Pandas")
st.sidebar.write("- NLTK VADER")
st.sidebar.write("- Streamlit")
st.sidebar.write("- Matplotlib")
st.sidebar.write("- Seaborn")

st.sidebar.divider()

st.sidebar.info(
    "This dashboard analyzes customer feedback "
    "and identifies positive, negative, and neutral reviews."
)
# Load Analyzed Dataset
df = pd.read_csv("Dataset/vader_analyzed_reviews.csv")

# Dashboard Title
st.title("📊 Flipkart Customer Review Sentiment Analysis")
st.write("Analyze customer reviews using NLP and VADER Sentiment Analysis.")

# Clean Rating Column
df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")

# Sentiment Counts
total_reviews = len(df)
positive_count = (df["vader_sentiment"] == "positive").sum()
negative_count = (df["vader_sentiment"] == "negative").sum()
neutral_count = (df["vader_sentiment"] == "neutral").sum()

# KPI Dashboard
st.subheader("📌 Sentiment Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Reviews", f"{total_reviews:,}")
col2.metric("Positive Reviews", f"{positive_count:,}")
col3.metric("Negative Reviews", f"{negative_count:,}")
col4.metric("Neutral Reviews", f"{neutral_count:,}")

# Total Products
st.metric("Total Products", f"{df['product_name'].nunique():,}")


# Average VADER Sentiment Score
average_score = df["vader_score"].mean()

st.metric(
    "Average VADER Score",
    f"{average_score:.3f}"
)

# Sentiment Score Interpretation
st.subheader("📌 Overall Sentiment Interpretation")

if average_score >= 0.05:
    st.success("Overall customer feedback is Positive 😊")
elif average_score <= -0.05:
    st.error("Overall customer feedback is Negative")
else:
    st.info("Overall customer feedback is Neutral")
    
# Customer Review Word Clouds
st.subheader("☁️ Customer Review Word Clouds")

col1, col2 = st.columns(2)

with col1:
    st.write("Positive Reviews")
    st.image(
        "Visualizations/positive_wordcloud.png",
        use_container_width=True
    )

with col2:
    st.write("Negative Reviews")
    st.image(
        "Visualizations/negative_wordcloud.png",
        use_container_width=True
    )    
# Sentiment Distribution
st.subheader("📊 Customer Sentiment Distribution")
st.bar_chart(df["vader_sentiment"].value_counts())

# Sentiment Percentage
st.subheader("📈 Sentiment Percentage")

sentiment_percentage = (
    df["vader_sentiment"].value_counts(normalize=True) * 100
).round(2)

st.bar_chart(sentiment_percentage)
st.dataframe(sentiment_percentage.rename("Percentage (%)"))

# Rating vs Sentiment
st.subheader("⭐ Customer Rating vs Sentiment")

rating_sentiment = pd.crosstab(
    df["Rate"],
    df["vader_sentiment"],
    normalize="index"
) * 100

rating_sentiment = rating_sentiment.round(2)

st.bar_chart(rating_sentiment)
st.dataframe(rating_sentiment, use_container_width=True)

# Product-Level Sentiment Analysis
st.subheader("🏆 Top 10 Products by Positive Sentiment")

product_df = df.groupby("product_name").agg(
    Total_Reviews=("vader_sentiment", "count"),
    Positive_Reviews=("vader_sentiment", lambda x: (x == "positive").sum()),
    Negative_Reviews=("vader_sentiment", lambda x: (x == "negative").sum())
).reset_index()

product_df["Positive_Percentage"] = (
    product_df["Positive_Reviews"] / product_df["Total_Reviews"] * 100
).round(2)

product_df["Negative_Percentage"] = (
    product_df["Negative_Reviews"] / product_df["Total_Reviews"] * 100
).round(2)

# Avoid products with very few reviews
product_df = product_df[product_df["Total_Reviews"] >= 20]

top_products = product_df.sort_values(
    by="Positive_Percentage",
    ascending=False
).head(10)

st.bar_chart(
    top_products.set_index("product_name")["Positive_Percentage"]
)

st.dataframe(top_products, use_container_width=True)

# Sentiment Filter
st.subheader("🔎 Explore Customer Reviews")

selected_sentiment = st.selectbox(
    "Select Sentiment",
    ["All", "positive", "negative", "neutral"]
)

if selected_sentiment == "All":
    filtered_df = df
else:
    filtered_df = df[
        df["vader_sentiment"] == selected_sentiment
    ]

st.write("Filtered Reviews:", len(filtered_df))

# Customer Rating Filter
st.subheader("⭐ Filter by Customer Rating")

rating_options = ["All"] + sorted(
    df["Rate"].dropna().unique().tolist()
)

selected_rating = st.selectbox(
    "Select Customer Rating",
    rating_options
)

if selected_rating != "All":
    filtered_df = filtered_df[
        filtered_df["Rate"] == selected_rating
    ]

# Customer Rating Distribution
st.subheader("⭐ Customer Rating Distribution")

rating_counts = df["Rate"].value_counts().sort_index()

st.bar_chart(rating_counts)

st.write("Rating-wise Review Counts:")
st.dataframe(
    rating_counts.rename("Number of Reviews")
)

# Top 10 Most Reviewed Products
st.subheader("📦 Top 10 Most Reviewed Products")

top_reviewed = product_df.sort_values(
    by="Total_Reviews",
    ascending=False
).head(10)

st.bar_chart(
    top_reviewed.set_index("product_name")["Total_Reviews"]
)

st.dataframe(top_reviewed, use_container_width=True)
# Product Search
st.subheader("🔍 Search Reviews by Product")

product_search = st.text_input("Enter Product Name")

if product_search:
    search_df = filtered_df[
        filtered_df["product_name"].str.contains(
            product_search,
            case=False,
            na=False
        )
    ]
else:
    search_df = filtered_df

st.write("Matching Reviews:", len(search_df))

st.dataframe(
    search_df[
        ["product_name", "Rate", "Review", "vader_sentiment"]
    ].head(100),
    use_container_width=True
)

# Top 10 Products by Negative Sentiment
st.subheader("⚠️ Top 10 Products with Highest Negative Sentiment")

top_negative = product_df.sort_values(
    by="Negative_Percentage",
    ascending=False
).head(10)

st.bar_chart(
    top_negative.set_index("product_name")["Negative_Percentage"]
)

st.dataframe(top_negative, use_container_width=True)

# Dataset Preview
st.subheader("Customer Review Data")
st.dataframe(df.head(10), use_container_width=True)


# Download Filtered Reviews
st.subheader("📥 Download Customer Reviews")

csv_data = search_df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="Download Reviews as CSV",
    data=csv_data,
    file_name="flipkart_filtered_reviews.csv",
    mime="text/csv"
)

# Key Business Insights
st.subheader("💡 Key Business Insights")

most_reviewed = product_df.loc[
    product_df["Total_Reviews"].idxmax()
]

most_positive = product_df.loc[
    product_df["Positive_Percentage"].idxmax()
]

most_negative = product_df.loc[
    product_df["Negative_Percentage"].idxmax()
]

st.write(
    f"📦 **Most Reviewed Product:** "
    f"{most_reviewed['product_name']} "
    f"({most_reviewed['Total_Reviews']} reviews)"
)

st.write(
    f"😊 **Highest Positive Sentiment:** "
    f"{most_positive['product_name']} "
    f"({most_positive['Positive_Percentage']}%)"
)

st.write(
    f"⚠️ **Highest Negative Sentiment:** "
    f"{most_negative['product_name']} "
    f"({most_negative['Negative_Percentage']}%)"
)

# Project Methodology
st.subheader("📚 Methodology")

st.write("""
1. Dataset collected from Flipkart customer reviews.
2. Removed duplicate records and reviews with missing text.
3. Applied VADER SentimentIntensityAnalyzer.
4. Classified reviews as Positive, Negative, or Neutral.
5. Analyzed sentiment by product and customer rating.
6. Created an interactive Streamlit dashboard.
""")

# Project Limitations
st.subheader("⚠️ Project Limitations")

st.write("""
- VADER is a lexicon-based sentiment analysis method.
- It may not accurately detect sarcasm or complex context.
- Some words may receive incorrect sentiment classifications.
- Results depend on the quality of customer review text.
""")