import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np
import networkx as nx
import folium
from streamlit_folium import st_folium

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv('movies.csv')

data = load_data()

# Title
st.title("Movie Dataset Visualizations")

# Sidebar for filtering options
st.sidebar.header("Filter Options")
year_range = st.sidebar.slider("Select Year Range", 
                              int(data['year'].min()), 
                              int(data['year'].max()), 
                              (1980, 2000))
selected_genre = st.sidebar.multiselect("Select Genre", 
                                        options=data['genre'].unique(), 
                                        default=data['genre'].unique())
min_runtime = st.sidebar.slider("Minimum Runtime (minutes)", 
                                 int(data['runtime'].min()), 
                                 int(data['runtime'].max()), 
                                 int(data['runtime'].min()))
min_score = st.sidebar.slider("Minimum Score", 
                               float(data['score'].min()), 
                               float(data['score'].max()), 
                               float(data['score'].min()))
selected_rating = st.sidebar.multiselect("Select Rating", 
                                         options=data['rating'].dropna().unique(), 
                                         default=data['rating'].dropna().unique())
selected_country = st.sidebar.multiselect("Select Country", 
                                          options=data['country'].dropna().unique(), 
                                          default=data['country'].dropna().unique())

# Filter data
filtered_data = data[(data['year'] >= year_range[0]) & 
                     (data['year'] <= year_range[1]) & 
                     (data['genre'].isin(selected_genre)) & 
                     (data['runtime'] >= min_runtime) & 
                     (data['score'] >= min_score) & 
                     (data['rating'].isin(selected_rating)) & 
                     (data['country'].isin(selected_country))]

# Visualization 1: Distribution of Scores
st.subheader("Distribution of Movie Scores 🎥")
with st.expander("Customize Histogram"):
    bins = st.slider("Number of Bins", min_value=5, max_value=50, value=20)
    color = st.color_picker("Select Bar Color", value='#1f77b4')
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(filtered_data['score'], bins=bins, kde=True, ax=ax, color=color)
ax.set_title("Score Distribution")
ax.set_xlabel("Score")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Visualization 2: Budget vs Gross Revenue vs Score
st.subheader("Budget vs Gross Revenue vs Score 💰")
with st.expander("Customize Scatter Plot"):
    scatter_palette = st.selectbox("Select Color Palette", options=['viridis', 'plasma', 'coolwarm', 'cubehelix'])
    point_size_range = st.slider("Point Size Range", 10, 200, (20, 100))
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=filtered_data, x='budget', y='gross', hue='score', size='score', 
                palette=scatter_palette, ax=ax, sizes=point_size_range)
ax.set_title("Budget vs Gross Revenue")
ax.set_xlabel("Budget")
ax.set_ylabel("Gross Revenue")
st.pyplot(fig)

# Visualization 3: Genre Distribution Over the Years
st.subheader("Genre Distribution Over the Years 📅")
with st.expander("Customize Line Chart"):
    line_style_dict = {
        'Solid': '-',
        'Dashed': '--',
        'Dash-dot': '-.',
        'Dotted': ':'
    }
    line_style_choice = st.selectbox("Line Style", options=list(line_style_dict.keys()))
    
fig, ax = plt.subplots(figsize=(10, 6))
genre_counts = filtered_data.groupby(['year', 'genre']).size().reset_index(name='count')
sns.lineplot(
    data=genre_counts, 
    x='year', 
    y='count', 
    hue='genre', 
    style='genre',
    markers=True,
    dashes=False,
    linestyle=line_style_dict[line_style_choice],
    ax=ax
)
ax.set_title("Genre Trends Over Years")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Movies")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig)

# Visualization 4: Runtime by Genre and Rating
st.subheader("Runtime by Genre and Rating ⏱️")
with st.expander("Customize Box Plot"):
    box_palette = st.selectbox("Boxplot Palette", options=['Set1', 'Set2', 'Set3', 'Spectral'])
fig, ax = plt.subplots(figsize=(12, 7))
sns.boxplot(data=filtered_data, x='genre', y='runtime', hue='rating', ax=ax, palette=box_palette)
ax.set_title("Runtime Distribution by Genre and Rating")
ax.set_xlabel("Genre")
ax.set_ylabel("Runtime (minutes)")
ax.legend(title="Rating")
st.pyplot(fig)

# Visualization 5: Word Cloud for Movie Titles
st.subheader("Word Cloud of Movie Titles ☁️")
selected_wordcloud_genre = st.sidebar.selectbox("Select Genre for Word Cloud", options=["All"] + list(data['genre'].dropna().unique()), index=0)
if selected_wordcloud_genre != "All":
    wordcloud_data = filtered_data[filtered_data['genre'] == selected_wordcloud_genre]
else:
    wordcloud_data = filtered_data

with st.expander("Customize Word Cloud"):
    wordcloud_width = st.slider("Word Cloud Width", 400, 1200, 800)
    wordcloud_height = st.slider("Word Cloud Height", 200, 800, 400)
    wordcloud_colormap = st.selectbox("Word Cloud Colormap", options=['Spectral', 'viridis', 'plasma', 'coolwarm'])

text = " ".join(wordcloud_data['name'].dropna().astype(str))
wordcloud = WordCloud(width=wordcloud_width, height=wordcloud_height, background_color='black', colormap=wordcloud_colormap).generate(text)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
ax.set_title(f"Word Cloud of Movie Titles ({selected_wordcloud_genre})")
st.pyplot(fig)

# Visualization 6: Interactive Scatter Plot with Plotly
st.subheader("Interactive Budget vs Gross Revenue 📊")
color_variable = st.sidebar.selectbox("Select Color Variable", options=['score', 'votes', 'runtime'], index=0)
size_variable = st.sidebar.selectbox("Select Size Variable", options=['votes', 'score', 'runtime'], index=0)
fig = px.scatter(filtered_data, x='budget', y='gross', color=color_variable, size=size_variable, 
                 hover_data=['name'], 
                 title="Interactive Budget vs Gross Revenue", 
                 labels={'budget': 'Budget', 'gross': 'Gross Revenue', color_variable: color_variable.title(), size_variable: size_variable.title()})
st.plotly_chart(fig)

# Visualization 7: Heatmap of Correlations
st.subheader("Heatmap of Numerical Correlations 🔥")
with st.expander("Customize Heatmap"):
    heatmap_cmap = st.selectbox("Heatmap Color Map", options=['coolwarm', 'viridis', 'plasma', 'magma'])
correlation_matrix = filtered_data[['score', 'votes', 'budget', 'gross', 'runtime']].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap=heatmap_cmap, fmt='.2f', ax=ax)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)

# Visualization 8: Top 10 Movies by Votes
st.subheader("Top 10 Movies by Votes 🏆")
with st.expander("Customize Bar Chart"):
    bar_color = st.color_picker("Select Bar Color", value='#2ca02c')
top_votes = filtered_data.nlargest(10, 'votes')[['name', 'votes']]
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=top_votes, x='votes', y='name', color=bar_color, ax=ax)
ax.set_title("Top 10 Movies by Votes")
ax.set_xlabel("Votes")
ax.set_ylabel("Movie Title")
st.pyplot(fig)

# Visualization 9: Pie Chart for Genre Distribution
st.subheader("Genre Distribution 🍰")
with st.expander("Customize Pie Chart"):
    pie_startangle = st.slider("Pie Chart Start Angle", 0, 360, 90)
genre_distribution = filtered_data['genre'].value_counts()
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(genre_distribution, labels=genre_distribution.index, autopct='%1.1f%%', startangle=pie_startangle, colors=sns.color_palette('Spectral', len(genre_distribution)))
ax.set_title("Genre Distribution Pie Chart")
st.pyplot(fig)

# Visualization 10: Clustering Movies with K-Means
st.subheader("Cluster Analysis of Movies 📍")
st.write("Clustering movies based on budget, gross revenue, and runtime.")
kmeans_data = filtered_data[['budget', 'gross', 'runtime']].dropna()
num_clusters = st.slider("Number of Clusters", 2, 10, 3)
kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(kmeans_data)
kmeans_data['Cluster'] = kmeans.labels_
fig = px.scatter_3d(kmeans_data, x='budget', y='gross', z='runtime', color='Cluster', 
                    title="3D Cluster Analysis", 
                    labels={'budget': 'Budget', 'gross': 'Gross Revenue', 'runtime': 'Runtime'})
st.plotly_chart(fig)

st.subheader("Popularity Trends of Famous Directors or Actors 🎬")
with st.expander("Pilih Filter"):  
    top_directors = data['director'].value_counts().nlargest(10).index
    selected_director = st.selectbox("Pilih Sutradara", options=top_directors)

trend_data = data[data['director'] == selected_director].groupby('year').size().reset_index(name='film_count')
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=trend_data, x='year', y='film_count', ax=ax)
ax.set_title(f"Jumlah Film oleh {selected_director} per Tahun")
ax.set_xlabel("Tahun")
ax.set_ylabel("Jumlah Film")
st.pyplot(fig)

st.subheader("Relative Profit Analysis By Genre 💵")
data['profit'] = data['gross'] - data['budget']
fig, ax = plt.subplots(figsize=(12, 7))
sns.boxplot(data=data, x='genre', y='profit', ax=ax, palette='Spectral')
ax.set_title("Keuntungan Relatif Berdasarkan Genre")
ax.set_xlabel("Genre")
ax.set_ylabel("Keuntungan")
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("Genre and Rating Interrelation Network 🕸️")
edge_data = data.groupby(['genre', 'rating']).size().reset_index(name='count')
G = nx.Graph()

for _, row in edge_data.iterrows():
    G.add_edge(row['genre'], row['rating'], weight=row['count'])

pos = nx.spring_layout(G)
fig, ax = plt.subplots(figsize=(12, 12))
nx.draw(G, pos, with_labels=True, node_size=3000, font_size=10, 
        node_color='skyblue', edge_color='gray', width=1, ax=ax)
st.pyplot(fig)

# 4. Distribusi Anggaran dan Keuntungan untuk Film Top Berdasarkan Skor
st.subheader("Budget and Profit Distribution for Top Films Based on Score 📈")
top_movies = data.nlargest(20, 'score')
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=top_movies, x='budget', y='profit', size='score', hue='score', palette='viridis', ax=ax)
ax.set_title("Anggaran vs Keuntungan (Top 20 Film berdasarkan Skor)")
ax.set_xlabel("Anggaran")
ax.set_ylabel("Keuntungan")
st.pyplot(fig)

# 5. Analisis Peringkat Berdasarkan Negara
st.subheader("Ranking Analysis by Country 🌍")
country_score = data.groupby('country')['score'].mean().reset_index()
fig = px.choropleth(country_score, locations='country', locationmode='country names', color='score', 
                    color_continuous_scale='viridis', title="Rata-rata Skor Film Berdasarkan Negara")
st.plotly_chart(fig)

# 6. Network Analysis: Aktor-Sutradara
st.subheader("Actor and Director Collaboration Network 🤝")
# Get top 10 directors by average movie score
top_directors = data.groupby('director')['score'].agg(['mean', 'count']).reset_index()
top_directors = top_directors[top_directors['count'] >= 3]  # Directors with at least 3 movies
top_directors = top_directors.nlargest(10, 'mean')

# Create visualization
fig, ax = plt.subplots(figsize=(12, 6))
bars = sns.barplot(data=top_directors, x='director', y='mean', 
                  palette='RdYlBu', ax=ax)
ax.set_title("Top 10 Directors by Average Movie Score")
ax.set_xlabel("Director")
ax.set_ylabel("Average Score")
plt.xticks(rotation=45, ha='right')

# Add count annotations
for i, v in enumerate(top_directors['count']):
    ax.text(i, ax.get_ylim()[0], f'n={v}', 
            horizontalalignment='center',
            verticalalignment='top')

st.pyplot(fig)

# Show raw data
st.subheader("Filtered Data Table 📋")
st.write(filtered_data)
