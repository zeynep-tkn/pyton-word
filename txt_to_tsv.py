import nltk
from nltk.corpus import words
import re

nltk.download('words')
english_vocab = set(words.words())

glove_file = 'glove.6B.100d.txt'  # ya da glove.6B.200d.txt hangisini kullanıyorsan
output_file = 'glove_english_10k.tsv'

top_n = 10000
count = 0

with open(glove_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as out:
    for line in f:
        values = line.strip().split()
        if len(values) < 10:
            continue  # yeterince boyut yoksa atla
        word = values[0]
        
        # Sadece İngilizce kelimeleri al
        word_cleaned = re.sub(r'[^a-zA-Z]', '', word).lower()
        if word_cleaned in english_vocab:
            vector = "\t".join(values[1:])
            out.write(f"{word}\t{vector}\n")
            count += 1
        
        if count >= top_n:
            break

print(f"{count} İngilizce kelime içeren vektör başarıyla kaydedildi: {output_file}")
