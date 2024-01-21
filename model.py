# Import necessary libraries
import numpy as np
import pandas as pd

# Read data from file and display a sample
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None)
print(df.sample(5))
print(df.shape)
print(df.info())

# Rename columns for clarity
df.columns = ['target', 'text']

# Encode the 'target' column (ham/spam) using LabelEncoder
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
print(df.head)

# Check for missing values and duplicates
print(df.isnull().sum())
print(df.duplicated().sum())
df = df.drop_duplicates(keep='first')
print(df.duplicated().sum())

# Exploratory Data Analysis (EDA)
print(df['target'].value_counts())

# Plot a pie chart to visualize the distribution of ham and spam messages
import matplotlib.pyplot as plt

plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct='%0.2f')
plt.show()

# Download NLTK data
import nltk

nltk.download('punkt')

# Create new columns for character count, word count, and sentence count
df['num_of_characters'] = df['text'].apply(len)
print(df.head())

df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
print(df.head())

df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
print(df.head())

# Display descriptive statistics for the new columns
print(df[['num_of_characters', 'num_words', 'num_sentences']].describe())
print(df[df['target'] == 0][['num_of_characters', 'num_words', 'num_sentences']].describe())
print(df[df['target'] == 1][['num_of_characters', 'num_words', 'num_sentences']].describe())

# Visualize character count distribution for ham and spam messages using histograms
import seaborn as sns

sns.histplot(df[df['target'] == 0]['num_of_characters'])
plt.show()
sns.histplot(df[df['target'] == 1]['num_of_characters'], color='red')
plt.show()

# Plot a heatmap to show correlation between numerical features
sns.heatmap(df.drop('text', axis=1).corr(), annot=True)
plt.show()

# Text preprocessing using stemming and removal of stopwords
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Apply the transform_text function to create a new column 'transformed_text'
df['transformed_text'] = df['text'].apply(transform_text)

# Display a sample of the transformed text
print(df.head())

# Generate word clouds for spam and ham messages
from wordcloud import WordCloud

wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15, 6))
plt.imshow(spam_wc)

ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15, 6))
plt.imshow(ham_wc)

# Analyze word frequency in spam messages
spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

from collections import Counter

sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0])
plt.xticks(rotation='vertical')
plt.show()

# Model Building

# Use CountVectorizer or TfidfVectorizer for text representation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)

# Apply vectorization to the transformed text
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Use different Naive Bayes classifiers for training and evaluation
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

# Fit the models, make predictions, and evaluate performance
gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print((precision_score(y_test, y_pred1)))

mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print((precision_score(y_test, y_pred2)))

bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
print((precision_score(y_test, y_pred3)))

# Save the trained model and vectorizer for future use
import pickle

pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))
