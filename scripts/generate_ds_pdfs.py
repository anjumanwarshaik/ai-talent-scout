# generate_ds_pdfs.py
import pandas as pd
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

# ---- Config ----
CSV_PATH = "data/raw/UpdatedResumeDataSet.csv"
OUTPUT_DIR = "data/raw/data/Data_Science"

def text_to_pdf(text, output_path):
    """Convert a resume text string into a proper text-based PDF."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch
    )
    styles = getSampleStyleSheet()
    story = []

    # Split text into paragraphs and add to PDF
    for para in text.split('\n'):
        para = para.strip()
        if para:
            story.append(Paragraph(para, styles['Normal']))
            story.append(Spacer(1, 6))

    doc.build(story)

def generate_ds_pdfs():
    # Load CSV
    df = pd.read_csv(CSV_PATH)
    ds_df = df[df["Category"] == 'Data Science'].reset_index(drop=True)
    print(f"Found {len(ds_df)} Data Science resumes")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success = 0
    failed = 0

    for i, row in ds_df.iterrows():
        output_path = os.path.join(OUTPUT_DIR, f"ds_resume_{i+1:03d}.pdf")
        try:
            text_to_pdf(row["Resume"],  output_path)
            print(f"Generated: ds_resume_{i+1:03d}.pdf")
            success += 1
        except Exception as e:
            print(f"Failed {i+1}: {e}")
            failed += 1

    print(f"\nDone! {success} PDFs generated, {failed} failed")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_ds_pdfs()