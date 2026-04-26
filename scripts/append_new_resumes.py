# append_new_resumes.py
import os
import pandas as pd
import numpy as np
from utils.pdf_reader import extract_text_from_pdf
from utils.preprocessing import clean_resume_text
from utils.embeddings import generate_embeddings
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# ---- Config: Only process this new folder ----
NEW_FOLDER = "Data_Science"
NEW_CATEGORY = "DATA-SCIENCE"

def append_new_resumes():
    csv_path = os.path.join(PROCESSED_DATA_DIR, "resumes_cleaned.csv")
    embed_path = os.path.join(PROCESSED_DATA_DIR, "resume_embeddings.npy")

    # 1. Load existing data
    print("Loading existing processed data...")
    existing_df = pd.read_csv(csv_path)
    existing_embeddings = np.load(embed_path)
    print(f"Existing resumes: {len(existing_df)} | Embeddings shape: {existing_embeddings.shape}")

    # 2. Process only the new 43 PDFs
    new_folder_path = os.path.join(RAW_DATA_DIR, NEW_FOLDER)
    new_texts = []

    print(f"\nProcessing new PDFs from: {new_folder_path}")
    for file in os.listdir(new_folder_path):
        if file.lower().endswith(".pdf"):
            raw_text = extract_text_from_pdf(os.path.join(new_folder_path, file))
            print(f"{file} -> extracted: {len(raw_text) if raw_text else 'FAILED'}")
            if raw_text:
                clean_text = clean_resume_text(raw_text)
                new_texts.append(clean_text)

    print(f"Successfully extracted text from {len(new_texts)} PDFs")

    # 3. Generate embeddings for only new resumes
    new_embeddings = generate_embeddings(new_texts)

    # 4. Append to existing CSV (only resume_text and category columns)
    new_df = pd.DataFrame({
        "resume_text": new_texts,
        "category": [NEW_CATEGORY] * len(new_texts)
    })
    combined_df = pd.concat([existing_df, new_df], axis=0, ignore_index=True)
    combined_df.to_csv(csv_path, index=False)
    print(f"\nCSV updated: {len(existing_df)} + {len(new_df)} = {len(combined_df)} resumes")

    # 5. Append to existing embeddings
    combined_embeddings = np.vstack([existing_embeddings, new_embeddings])
    np.save(embed_path, combined_embeddings)
    print(f"Embeddings updated: {existing_embeddings.shape} -> {combined_embeddings.shape}")

    print("\nDone! Your existing work is preserved and new resumes are appended.")

if __name__ == "__main__":
    append_new_resumes()