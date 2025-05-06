from datasets import load_dataset

# Dataset'i yükle
dataset = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset")

# Eğitim ve test verilerine erişim
train_data = dataset['train']
test_data = dataset['test']

# İlk örneğe göz at
print(train_data[0])
