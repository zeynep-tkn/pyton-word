import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from dataset_kucultme import embedding_index

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Veri setini oku
train_df = pd.read_csv("data/train.csv")
train_df = train_df.dropna(subset=["text"])

# Tokenleştir, temizle ve lemmatize et
def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return filtered

train_df["tokens"] = train_df["text"].apply(preprocess)

# TF-IDF hesapla
tfidf = TfidfVectorizer(
    tokenizer=lambda x: x,       # Kendi token listen var (önceden tokenize edilmiş)
    lowercase=False,             # Tokenizer kendininkini kullandığın için lowercase işlemi gereksiz
    token_pattern=None           # Buraya özellikle `None` yazarsan uyarı kalkar
)

tfidf_matrix = tfidf.fit_transform(train_df["tokens"])

# TF-IDF ağırlıklı ortalama embedding
def weighted_embedding(tokens):
    vec = np.zeros(100)
    total_weight = 0
    for word in tokens:
        if word in embedding_index and word in tfidf.vocabulary_:
            weight = tfidf.idf_[tfidf.vocabulary_[word]]
            vec += embedding_index[word] * weight
            total_weight += weight
    return vec / total_weight if total_weight != 0 else vec

train_df["embedding"] = train_df["tokens"].apply(weighted_embedding)

X = np.stack(train_df["embedding"].values)
y = train_df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Daha güçlü model: Random Forest
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Geliştirilmiş Doğruluk (accuracy):", accuracy)
