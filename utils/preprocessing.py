# utils/preprocessing.py
import re
import string

def clean_resume_text(text: str) -> str:
    """
    Cleans raw resume text by removing URLs, special characters, 
    and fixing whitespaces.
    """
    if not text:
        return ""
    
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+\s*', ' ', text)

    # 3. Remove Emails
    text = re.sub(r'\S*@\S*\s?', ' ', text)

    # 4. Remove special characters and punctuation
    # This keeps only alphanumeric characters and basic spacing
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)

    # 5. Remove non-ascii characters (like icons/bullets)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)

    # 6. Remove extra whitespace/newlines
    text = re.sub(r'\s+', ' ', text).strip()

    return text