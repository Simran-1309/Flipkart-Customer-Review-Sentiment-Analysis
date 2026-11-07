# Flipkart Customer Review Sentiment Analysis

## Project Overview
This project analyzes Flipkart customer reviews using Natural Language Processing (NLP) and VADER Sentiment Analysis.

The objective is to understand customer opinions, identify positive and negative feedback, analyze product-level sentiment, and present insights through an interactive Streamlit dashboard.

Dataset

The dataset used in this project is the Flipkart Product Customer Reviews Dataset.

Source: Download from Kaggle

Dataset Setup

Download Dataset-SA.csv from Kaggle.

Create a folder named Dataset in the project directory.

Place Dataset-SA.csv inside the Dataset folder.

Run sentiment_analysis.py to generate the cleaned and VADER-analyzed CSV files.

Run app.py to launch the dashboard.

Note: Dataset files are excluded from this GitHub repository because of their large size. Users must download the dataset separately.

## Technologies Used
- Python
- Pandas
- NLTK (VADER)
- Matplotlib
- Seaborn
- WordCloud
- Streamlit

## Project Workflow
1. Dataset collection and exploration.
2. Data cleaning and duplicate removal.
3. Review text processing.
4. Sentiment scoring using VADER.
5. Classification into Positive, Negative, and Neutral.
6. Product-level and rating-wise sentiment analysis.
7. Interactive dashboard development using Streamlit.

## Key Results

| Metric | Result |
|---|---:|
| Total Reviews | 154,475 |
| Positive Reviews | 113,226 |
| Neutral Reviews | 26,771 |
| Negative Reviews | 14,478 |
| Positive Sentiment | 73.30% |
| Neutral Sentiment | 17.33% |
| Negative Sentiment | 9.37% |
| Average VADER Score | 0.329 |

## Dashboard Features
- Sentiment distribution analysis
- Sentiment percentage visualization
- Customer rating analysis
- Top 10 positive products
- Top 10 negative products
- Top 10 most-reviewed products
- Sentiment and rating filters
- Product name search
- Positive and negative word clouds
- Key business insights
- Filtered review CSV download

## How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/Simran-1309/Flipkart-Customer-Review-Sentiment-Analysis.git
```

### 2. Open the project folder
```bash
cd Flipkart-Customer-Review-Sentiment-Analysis
```

### 3. Install dependencies
```bash
python -m pip install pandas matplotlib seaborn nltk streamlit wordcloud
```

### 4. Download VADER lexicon
```bash
python -m nltk.downloader vader_lexicon
```

### 5. Run the dashboard
```bash
python -m streamlit run app.py
```

## Project Limitations
- VADER is a lexicon-based sentiment analysis technique.
- It may struggle with sarcasm and complex sentence context.
- Sentiment predictions depend on the words used in customer reviews.

## Conclusion
This project demonstrates how NLP and VADER can be used to analyze customer feedback, identify product sentiment patterns, and generate useful business insights through an interactive dashboard.