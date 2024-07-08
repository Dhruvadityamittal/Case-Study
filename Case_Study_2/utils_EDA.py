# Utils file for EDA
from wordcloud import WordCloud,STOPWORDS
from matplotlib.pylab import plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def get_label_dist(df):
    fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)
    plt.tight_layout()

    # Plot the pie chart with percentages
    df.groupby('Y').count()['Id'].plot(
        kind='pie',
        ax=axes[0],
        labels=['High Quality', 'Low Quality', 'Low Quality (Multiple Edits)'],
        autopct='%1.1f%%'  # This will add the percentage annotations
    )

    # Plot the countplot
    sns.countplot(x=df['Y'], hue=df['Y'], ax=axes[1])

    # Annotate the counts inside the bars
    for p in axes[1].patches:
        axes[1].annotate(
            format(p.get_height(), '.0f'),  # Format the count as an integer
            (p.get_x() + p.get_width() / 2., p.get_height() / 2.),  # Position the text
            ha='center', va='center',  # Center align the text
            fontsize=12, color='white', weight='bold'  # Font settings
        )

    # Adjust y-axis label for both plots
    axes[0].set_ylabel('')
    axes[1].set_ylabel('')

    # Set x-axis tick labels for countplot
    axes[1].set_xticklabels(['High Quality', 'Low Quality (Closed)', 'Low Quality (Multiple Edits)'])

    # Adjust tick parameters
    axes[0].tick_params(axis='x', labelsize=15)
    axes[0].tick_params(axis='y', labelsize=15)
    axes[1].tick_params(axis='x', labelsize=15)
    axes[1].tick_params(axis='y', labelsize=15)

    # Set titles for both plots
    axes[0].set_title('Label Distribution in Training Set', fontsize=13)
    axes[1].set_title('Label Count in Training Set', fontsize=13)

    plt.show()
def plot_wordCloud(df, label):
    plt.figure(figsize = (20,20)) # Text that is of high quality
    wc = WordCloud(max_words = 1000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df[df['class'] == label].text))
    plt.imshow(wc , interpolation = 'bilinear')

def show_donut_plot(df,col):
  """
  Creates a donut plot showing the distribution of data in a specific column.
  """
  rating_data = df.groupby(col)[['Id']].count().head(10)

  # Extract the counts as a 1D array
  counts = rating_data['Id'].to_numpy()

  plt.figure(figsize=(8, 8))
  plt.pie(counts, autopct='%1.0f%%', startangle=140, pctdistance=1.1, shadow=True)

  # Create a center circle for aesthetics
  gap = plt.Circle((0, 0), 0.5, fc='white')
  fig = plt.gcf()
  fig.gca().add_artist(gap)

  plt.axis('equal')

  # Extract labels from the index of rating_data
  cols = rating_data.index.to_numpy()
  plt.legend(cols)

  plt.title('Donut Plot: SOF Questions by ' + str(col), loc='center')

  plt.show()


def get_character_dist(df):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
    fig.suptitle('Characters in Texts', fontsize=16, weight='bold')

    # Plot for High Quality
    text_len = df[df['class'] == 0]['text'].str.len()
    ax1.hist(text_len, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
    ax1.set_title('High Quality', fontsize=14)
    ax1.set_xlabel('Text Length', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(0, 7000)

    # Plot for Low Quality (closed)
    text_len = df[df['class'] == 1]['text'].str.len()
    ax2.hist(text_len, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    ax2.set_title('Low Quality (Edited)', fontsize=14)
    ax2.set_xlabel('Text Length', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlim(0, 7000)

    # Plot for Low Quality (open)
    text_len = df[df['class'] == 2]['text'].str.len()
    ax3.hist(text_len, bins=20, color='lightskyblue', edgecolor='black', alpha=0.7)
    ax3.set_title('Low Quality (Open)', fontsize=14)
    ax3.set_xlabel('Text Length', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlim(0, 7000)

    # Adjust the layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()
def get_word_dist(df):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
    fig.suptitle('Average Word Length in Each Text', fontsize=16, weight='bold')

    # Plot for High Quality
    word = df[df['class'] == 0]['text'].str.split().apply(lambda x: [len(i) for i in x])
    mean_word_length = word.map(lambda x: np.mean(x))
    sns.kdeplot(mean_word_length, ax=ax1, color='red', fill=True)
    mean_value = mean_word_length.mean()
    ax1.axvline(mean_value, color='red', linestyle='dashed', linewidth=1)
    ax1.annotate(f'Mean: {mean_value:.2f}', xy=(mean_value, ax1.get_ylim()[1]*0.9), 
                xytext=(mean_value+0.5, ax1.get_ylim()[1]*0.9), 
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12, color='black')
    ax1.set_title('High Quality', fontsize=14)
    ax1.set_xlabel('Average Word Length', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(0, 20)

    # Plot for Low Quality (closed)
    word = df[df['class'] == 1]['text'].str.split().apply(lambda x: [len(i) for i in x])
    mean_word_length = word.map(lambda x: np.mean(x))
    sns.kdeplot(mean_word_length, ax=ax2, color='green', fill=True)
    mean_value = mean_word_length.mean()
    ax2.axvline(mean_value, color='red', linestyle='dashed', linewidth=1)
    ax2.annotate(f'Mean: {mean_value:.2f}', xy=(mean_value, ax2.get_ylim()[1]*0.9), 
                xytext=(mean_value+0.5, ax2.get_ylim()[1]*0.9), 
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12, color='black')
    ax2.set_title('Low Quality (Closed)', fontsize=14)
    ax2.set_xlabel('Average Word Length', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlim(0, 20)

    # Plot for Low Quality (open)
    word = df[df['class'] == 2]['text'].str.split().apply(lambda x: [len(i) for i in x])
    mean_word_length = word.map(lambda x: np.mean(x))
    sns.kdeplot(mean_word_length, ax=ax3, color='blue', fill=True)
    mean_value = mean_word_length.mean()
    ax3.axvline(mean_value, color='red', linestyle='dashed', linewidth=1)
    ax3.annotate(f'Mean: {mean_value:.2f}', xy=(mean_value, ax3.get_ylim()[1]*0.9), 
                xytext=(mean_value+0.5, ax3.get_ylim()[1]*0.9), 
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12, color='black')
    ax3.set_title('Low Quality (Open)', fontsize=14)
    ax3.set_xlabel('Average Word Length', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_xlim(0, 20)

    # Adjust the layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

def get_top_text_ngrams(corpus, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def plt_n_gram_dist(df,n):
    plt.figure(figsize = (16,5))
    gram_mapping = {1:"Unigram", 2:"Bigram", 3:"Trigram"}
    plt.title(gram_mapping[n])
    most_common_uni = get_top_text_ngrams(df.text,10,n)
    most_common_uni = dict(most_common_uni)
    sns.set_palette("husl")
    sns.barplot(x=list(most_common_uni.values()),y=list(most_common_uni.keys()))