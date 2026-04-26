#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Load your new CSV file
df = pd.read_csv("candidates.csv")

# 2. Extract the text you want the AI to read
# IMPORTANT: Change 'resume_text' to the actual name of the column in your CSV
# that contains the candidate's details (e.g., 'background', 'skills', or 'summary')
texts = df['resume_text'].fillna("").tolist()

# 3. Load a standard, fast SentenceTransformer model
print("Loading AI model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 4. Generate the embeddings
print("Generating embeddings... (this might take a minute)")
embeddings = model.encode(texts)

# 5. Save the file
np.save("embeddings.npy", embeddings)
print("Success! embeddings.npy has been created.")

