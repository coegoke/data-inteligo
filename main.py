import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import streamlit as st
st.set_page_config(
    page_title="Twitter Sentiment Analyzer", page_icon="📊", layout="wide"
)
st.title(":bar_chart: Review Sentiment Dashboard")
def get_top_n_gram(tweet_df, ngram_range, n=10):
    corpus = tweet_df["Text"]
    vectorizer = CountVectorizer(
        analyzer="word", ngram_range=ngram_range
    )
    X = vectorizer.fit_transform(corpus.astype(str).values)
    words = vectorizer.get_feature_names_out()
    words_count = np.ravel(X.sum(axis=0))
    df = pd.DataFrame(zip(words, words_count))
    df.columns = ["words", "counts"]
    df = df.sort_values(by="counts", ascending=False).head(n)
    df["words"] = df["words"].str.title()
    return df

def plot_n_gram(n_gram_df, title, color="#54A24B"):
    fig = px.bar(
        x=n_gram_df.counts,
        y=n_gram_df.words,
        title="<b>{}</b>".format(title),
        text_auto=True,
    )
    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(title=None)
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Count=%{x}", marker_color=color)
    return fig

def plot_sentiment(tweet_df):
    sentiment_count = tweet_df["status"].value_counts()
    fig = px.pie(
        values=sentiment_count.values,
        names=sentiment_count.index,
        hole=0.3,
        title="<b>Sentiment Distribution</b>",
        color=sentiment_count.index,
        color_discrete_map={"Positive": "#1F77B4", "Negative": "#FF7F0E"},
    )
    fig.update_traces(
        textposition="inside",
        texttemplate="%{label}<br>%{value} (%{percent})",
        hovertemplate="<b>%{label}</b><br>Percentage=%{percent}<br>Count=%{value}",
    )
    fig.update_layout(showlegend=False)
    return fig



adjust_top_pad = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
    """
st.markdown(adjust_top_pad, unsafe_allow_html=True)
df = pd.read_csv('data_deploy.csv')
st.header("Please Filter Here:")
sumber = st.multiselect(
    "Select the sumber:",
    options=df["sumber"].unique(),
    default=['twitter','playstore']
)

status = st.multiselect(
    "Select the status:",
    options=df["status"].unique(),
    default=df["status"].unique()
)

df_selection = df.query(
    "sumber == @sumber & status ==@status"
)
def make_dashboard(tweet_df, bar_color, wc_color):
        # first row
        col1, col2, col3 = st.columns([25, 40, 35])
        with col1:
            sentiment_plot = plot_sentiment(tweet_df)
            sentiment_plot.update_layout(title_x=0.5,title_font=dict(size=24))
            st.plotly_chart(sentiment_plot, theme=None, use_container_width=True)
        with col2:
            top_unigram = get_top_n_gram(tweet_df, ngram_range=(1, 1), n=10)
            unigram_plot = plot_n_gram(
                top_unigram, title="Top 10 Occuring Words", color=bar_color
            )
            unigram_plot.update_layout(title_font=dict(size=24))
            st.plotly_chart(unigram_plot, theme=None, use_container_width=True)
        with col3:
            top_bigram = get_top_n_gram(tweet_df, ngram_range=(2, 2), n=10)
            bigram_plot = plot_n_gram(
                top_bigram, title="Top 10 Occuring Bigrams", color=bar_color
            )
            bigram_plot.update_layout(title_font=dict(size=24))
            st.plotly_chart(bigram_plot, theme=None, use_container_width=True)
make_dashboard(df_selection, bar_color="#54A24B", wc_color="Greens")
value_counts = df_selection['Keywords_kategori'].value_counts().head(8).reset_index()
value_counts.columns = ['Category', 'Count']
value_counts = value_counts.sort_values(by='Count', ascending=True)
fig_topic = px.bar(value_counts, y='Category', x='Count', orientation='h', title='top Topic Modelling')
st.plotly_chart(fig_topic, use_container_width=True)

