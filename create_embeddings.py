import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv("medicine_dataset.csv")
df['usage_embeddings'] = df['medicine_usage'].apply(lambda x: model.encode(x))

df.to_pickle('medicine_embeddings.pkl')
