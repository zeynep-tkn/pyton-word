import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download("punkt")
nltk.download("stopwords")

# Embedding yolunu belirt (örneğin: 'glove.6B.100d.txt')
embedding_path = "glove.6B.100d.txt"

# Veriyi yükle
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Tüm verilerdeki kelimeleri topla
all_words = set()
for text in pd.concat([train_df["text"], test_df["text"]]):
    if isinstance(text, str):  # Boş olmayan satırları kontrol et
        tokens = word_tokenize(text.lower())
        all_words.update(tokens)

print(f"Toplam benzersiz kelime sayısı: {len(all_words)}")

# GloVe embedding sadece bu kelimeler için yüklenecek
embedding_index = {}
with open(embedding_path, encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        if word in all_words:
            vector = np.asarray(values[1:], dtype="float32")
            embedding_index[word] = vector

print(f"Yüklenen kelime sayısı: {len(embedding_index)}")

# İstenirse bu küçük embedding sözlüğü kaydedilebilir
# Örnek: np.save("kucuk_embedding.npy", embedding_index)
