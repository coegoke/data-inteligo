import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import unicodedata
import re
from nltk.util import ngrams
from nltk.corpus import stopwords
st.set_page_config(page_title="Review Dashboard", page_icon=":shark:", layout="wide")

df = pd.read_csv('data_deploy.csv')

# ---- SIDEBAR ----
st.sidebar.header("Please Filter Here:")
sumber = st.sidebar.multiselect(
    "Select the sumber:",
    options=df["sumber"].unique(),
    default=['twitter','playstore']
)

status = st.sidebar.multiselect(
    "Select the status:",
    options=df["status"].unique(),
    default=df["status"].unique()
)

df_selection = df.query(
    "sumber == @sumber & status ==@status"
)

# Join the different processed titles together.
# ---- MAINPAGE ----
st.title(":bar_chart: Review Dashboard")
st.markdown("##")
st.set_option('deprecation.showPyplotGlobalUse', False)
value_counts = df_selection['Keywords_kategori'].value_counts().head().reset_index()
value_counts.columns = ['Category', 'Count']
value_counts = value_counts.sort_values(by='Count', ascending=True)
fig_topic = px.bar(value_counts, y='Category', x='Count', orientation='h', title='Grafik Batang Vertikal Berdasarkan Value Counts')
long_string = ','.join(list(df_selection['Text'].astype(str).values))
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
WordCloud = wordcloud.generate(long_string)
# plt.imshow(WordCloud)
# st.pyplot()
 # Visualize the word cloud
# Convert WordCloud to a Matplotlib figure
wordcloud_matplotlib = plt.figure()
plt.imshow(WordCloud)
plt.axis("off")  # Turn off axis
plt.title("Word Cloud")

left_column, right_column = st.columns([60,40])
left_column.plotly_chart(fig_topic, use_container_width=True)
right_column.pyplot(wordcloud_matplotlib, use_container_width=True)

def basic_clean(text):
  text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  return words
words = basic_clean(''.join(str(df_selection['Text'].tolist())))
onegrams = (pd.Series(nltk.ngrams(words,1)).value_counts()).head(10)
bigrams = (pd.Series(nltk.ngrams(words,2)).value_counts()).head(10)
trigrams = (pd.Series(nltk.ngrams(words,3)).value_counts()).head(10)

df_grams1 = pd.Series(onegrams).reset_index()
df_grams1['Pasangan Kata'] = df_grams1['index'].apply(lambda x: ' '.join(x))  # Mengubah tuple menjadi string
# Mengurutkan DataFrame berdasarkan frekuensi
df_grams1 = df_grams1.rename(columns={0: 'Jumlah'})
df_grams1 = df_grams1.sort_values(by='Jumlah', ascending=False)
# Membuat bar chart interaktif menggunakan Plotly
fig_onegrams = px.bar(df_grams1, x='Jumlah', y='Pasangan Kata', orientation='h', 
             title='one word Phrases')
fig_onegrams.update_traces(marker_color='skyblue')
fig_onegrams.update_layout(xaxis_title='Jumlah', yaxis_title='One word Phrases', yaxis_categoryorder='total ascending')

df_grams2 = pd.Series(bigrams).reset_index()
df_grams2['Pasangan Kata'] = df_grams2['index'].apply(lambda x: ' '.join(x))  # Mengubah tuple menjadi string
# Mengurutkan DataFrame berdasarkan frekuensi
df_grams2 = df_grams2.rename(columns={0: 'Jumlah'})
df_grams2 = df_grams2.sort_values(by='Jumlah', ascending=False)
# Membuat bar chart interaktif menggunakan Plotly
fig_bigrams = px.bar(df_grams2, x='Jumlah', y='Pasangan Kata', orientation='h', 
             title='Two word Phrases')
fig_bigrams.update_traces(marker_color='skyblue')
fig_bigrams.update_layout(xaxis_title='Jumlah', yaxis_title='Two word Phrases', yaxis_categoryorder='total ascending')

df_grams3 = pd.Series(trigrams).reset_index()
df_grams3['Pasangan Kata'] = df_grams3['index'].apply(lambda x: ' '.join(x))  # Mengubah tuple menjadi string
# Mengurutkan DataFrame berdasarkan frekuensi
df_grams3 = df_grams3.rename(columns={0: 'Jumlah'})
df_grams3 = df_grams3.sort_values(by='Jumlah', ascending=False)
# Membuat bar chart interaktif menggunakan Plotly
fig_trigrams = px.bar(df_grams3, x='Jumlah', y='Pasangan Kata', orientation='h', 
             title='Three word Phrases')
fig_trigrams.update_traces(marker_color='skyblue')
fig_trigrams.update_layout(xaxis_title='Jumlah', yaxis_title='Three word Phrases', yaxis_categoryorder='total ascending')


left_column2,center_column2, right_column2 = st.columns(3)
left_column2.plotly_chart(fig_onegrams, use_container_width=True)
center_column2.plotly_chart(fig_bigrams, use_container_width=True)
right_column2.plotly_chart(fig_trigrams, use_container_width=True)