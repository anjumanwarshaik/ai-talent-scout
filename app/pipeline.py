# app/pipeline.py
import os
import pandas as pd
import numpy as np
from utils.pdf_reader import extract_text_from_pdf
from utils.preprocessing import clean_resume_text
from utils.embeddings import generate_embeddings
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def run_extraction_pipeline():
    all_text = []
    all_labels = []

    print(f"Scanning directory: {RAW_DATA_DIR}")

    for folder in os.listdir(RAW_DATA_DIR):
        folder_path = os.path.join(RAW_DATA_DIR, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith(".pdf"):
                    raw_text = extract_text_from_pdf(os.path.join(folder_path, file))
                    
                    if raw_text:
                        # Clean the text here
                        clean_text = clean_resume_text(raw_text)
                        
                        all_text.append(clean_text)
                        all_labels.append(folder)

    # 1. Save the Cleaned Text to CSV
    df = pd.DataFrame({"resume_text": all_text, "category": all_labels})
    csv_path = os.path.join(PROCESSED_DATA_DIR, "resumes_cleaned.csv")
    df.to_csv(csv_path, index=False)

    # 2. Generate Embeddings (The AI Part)
    embeddings = generate_embeddings(df["resume_text"].tolist())

    # 3. Save Embeddings as Numpy file
    embed_path = os.path.join(PROCESSED_DATA_DIR, "resume_embeddings.npy")
    np.save(embed_path, embeddings)

    print(f"Success! Saved {len(df)} resumes and thier embeddings.")
    return df, embeddings