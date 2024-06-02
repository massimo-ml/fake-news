import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DataPreprocessingBeforeClassifiers:
    def __init__(self, df_train_path: str, df_test_path: str, nltk_data_path: str) -> None:
        nltk.data.path.append(nltk_data_path)
        self.df_train = pd.read_csv(df_train_path) 
        self.df_test = pd.read_csv(df_test_path)
        
        self.df_train['title'] = self.df_train['title'].apply(self.clean_text)
        self.df_train['text'] = self.df_train['text'].apply(self.clean_text)
        self.df_test['title'] = self.df_test['title'].apply(self.clean_text)
        self.df_test['text'] = self.df_test['text'].apply(self.clean_text)

        self.df_train['title'] = self.df_train['title'].apply(word_tokenize)
        self.df_train['text'] = self.df_train['text'].apply(word_tokenize)
        self.df_test['title'] = self.df_test['title'].apply(word_tokenize)
        self.df_test['text'] = self.df_test['text'].apply(word_tokenize)

        stop_words = set(stopwords.words('english'))

        self.df_train['title'] = self.df_train['title'].apply(lambda x: [word for word in x if word not in stop_words])
        self.df_train['text'] = self.df_train['text'].apply(lambda x: [word for word in x if word not in stop_words])
        self.df_test['title'] = self.df_test['title'].apply(lambda x: [word for word in x if word not in stop_words])
        self.df_test['text'] = self.df_test['text'].apply(lambda x: [word for word in x if word not in stop_words])

        lemmatizer = WordNetLemmatizer()

        self.df_train['title'] = self.df_train['title'].apply(lambda x: [lemmatizer.lemmatize(token) for token in x])
        self.df_train['text'] = self.df_train['text'].apply(lambda x: [lemmatizer.lemmatize(token) for token in x])
        self.df_test['title'] = self.df_test['title'].apply(lambda x: [lemmatizer.lemmatize(token) for token in x])
        self.df_test['text'] = self.df_test['text'].apply(lambda x: [lemmatizer.lemmatize(token) for token in x])

        self.df_train['joined'] = self.df_train['title'].apply(lambda x: ' '.join(x)) + ' ' + self.df_train['text'].apply(lambda x: ' '.join(x))
        self.df_test['joined'] = self.df_test['title'].apply(lambda x: ' '.join(x)) + ' ' + self.df_test['text'].apply(lambda x: ' '.join(x))

        self.X_train = self.df_train['joined']
        self.y_train = self.df_train['label']
        self.X_test = self.df_test['joined']
        self.y_test = self.df_test['label']

    def clean_text(self, text: str) -> str:
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters and numbers
        text = text.lower()  # Convert to lowercase
        return text
    
    def out_ml(self):
        vectorizer = TfidfVectorizer(max_features=10000)
        X_train_tfidf = vectorizer.fit_transform(self.X_train)
        X_test_tfidf = vectorizer.transform(self.X_test)
        return X_train_tfidf, self.y_train, X_test_tfidf, self.y_test
    
    def out_dl(self):
        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(self.X_train)
        X_train_seq = tokenizer.texts_to_sequences(self.X_train)
        X_test_seq = tokenizer.texts_to_sequences(self.X_test)
        maxlen = 500
        X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
        X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)
        return X_train_pad, self.y_train, X_test_pad, self.y_test