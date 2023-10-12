import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import streamlit as st
st.set_page_config(
    page_title="Review Sentiment Dashboard", page_icon="📊", layout="wide"
)
page_bg_img="""
<style>
[data-testid="stAppViewContainer"] > .main{
background-image: url("https://i.pinimg.com/564x/26/3c/93/263c93e03a3d1c3de47dd7c882f93cc8.jpg");
background-size:100%;
background-position:top;
background-repeat:no-repeat;
background-attachment:local;
background-color: black !important;
}
[data-testid="stHeader"]{
background-color: rgba(0,0,0,0);
}
"""
st.markdown(page_bg_img,unsafe_allow_html=True)
adjust_top_pad = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
    """
st.markdown(adjust_top_pad, unsafe_allow_html=True)
st.title(":bar_chart: Review Sentiment Dashboard")
df = pd.read_csv('data_deploy.csv')
atas2, atas3,atas1 = st.columns([25,30,45])
with atas2:
    sumber = st.multiselect(
        "Pilih Sumber:",
        options=df["sumber"].unique(),
        default=['twitter','playstore']
    )
with atas3:
    status = st.multiselect(
        "Pilih Status:",
        options=df["status"].unique(),
        default=df["status"].unique()
    )

df_selection = df.query(
    "sumber == @sumber & status ==@status"
    
)
with atas1:
     
# Define the number of reviews
    num_reviews = len(df_selection)
    # with open('style.css') as f:
    #     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
 
    # Create a header with right-aligned text
    st.markdown("<div style='text-align: right;font-size: 36px;font-weight: bold;'>{} Review</div>".format(num_reviews), unsafe_allow_html=True)


def get_top_n_gram(tweet_df, ngram_range, n=10):
    corpus = tweet_df["Text_clean"]
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
    fig.update_layout(plot_bgcolor="rgba(0, 0, 0, 0)", paper_bgcolor='rgba(0, 0, 0, 0)',margin=dict( r=0))
    fig.update_xaxes(title=None)
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Count=%{x}", marker_color=color, textfont=dict(color='white'))
    return fig

def plot_sentiment(tweet_df):
    sentiment_count = tweet_df["status"].value_counts()
    fig = px.pie(
        values=sentiment_count.values,
        names=sentiment_count.index,
        hole=0.3,
        title="<b>Sentiment Distribution</b>",
        color=sentiment_count.index,
        color_discrete_map={"positif": "#54A24B", "negatif": "red", 'netral':'yellow'},
    )
    fig.update_traces(
        textposition="inside",
        texttemplate="%{label}<br>%{value} (%{percent})",
        hovertemplate="<b>%{label}</b><br>Percentage=%{percent}<br>Count=%{value}",
    )
    fig.update_layout(showlegend=False,margin=dict( l=0, r=0),plot_bgcolor="rgba(0, 0, 0, 0)", paper_bgcolor='rgba(0, 0, 0, 0)')
    return fig



adjust_top_pad = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
    """
st.markdown(adjust_top_pad, unsafe_allow_html=True)

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
col21, col22 = st.columns([50,50])
with col21:
    value_counts = df_selection['Keywords_kategori'].value_counts().head(8).reset_index()
    value_counts.columns = ['Category', 'Count']
    value_counts = value_counts.sort_values(by='Count', ascending=True)
    fig_topic = px.bar(value_counts, y='Category', x='Count', orientation='h', title='Top Topic Modelling')
    fig_topic.update_layout(title_font=dict(size=24),plot_bgcolor="rgba(0, 0, 0, 0)", paper_bgcolor='rgba(0, 0, 0, 0)')
    fig_topic.update_yaxes(tickfont=dict(color='black'),title_text=None)
    fig_topic.update_xaxes(tickfont=dict(color='black'),showline=True, title_text=None,showgrid=True, gridwidth=1, gridcolor='white')
    fig_topic.update_traces(hovertemplate="<b>%{y}</b><br>Count=%{x}", marker_color='#54A24B', textfont=dict(color='white'), text=value_counts['Count'])
    st.plotly_chart(fig_topic, use_container_width=True)
with col22:
    def sentiment_color(sentiment):
            if sentiment == "positif":
                return "background-color: green; color: white"
            elif sentiment =='netral':
                return "background-color: yellow"
            else:
                return "background-color: red; color: white"
    df_random = df_selection.sample(frac=1).reset_index()
    st.dataframe(
    df_random[["status", "Text_clean"]].style.applymap(
                    sentiment_color, subset=["status"]
                ))
