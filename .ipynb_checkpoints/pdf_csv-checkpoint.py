import pdfplumber
import pandas as pd
import re

def extract_text_from_pdf(pdf_path):
    data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines = text.split("\n")
                data.extend(lines)
    return data

def clean_and_structure_data(data):
    structured_data = []
    temp_row = []
    for line in data:
        if re.match(r'^\s*$', line):  # Skip empty lines
            continue
        if is_new_record(line):  # Check if a new record starts
            if temp_row:
                structured_data.append(temp_row)
            temp_row = [line]
        else:
            temp_row.append(line)  # Append to the previous record
    
    if temp_row:  # Add last record
        structured_data.append(temp_row)
    
    return structured_data

def is_new_record(line):
    # Adjust this condition based on how new categories are identified in your PDF
    return bool(re.match(r'^[A-Z]+', line))  # Example: Categories start with uppercase letters

def save_to_csv(data, output_csv):
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, header=False)

# Jupyter Notebook does not require __name__ == "__main__"
pdf_path = "input1.pdf"  # Change this to your PDF file path
output_csv = "pdf_csv.csv"

raw_data = extract_text_from_pdf(pdf_path)
structured_data = clean_and_structure_data(raw_data)
save_to_csv(structured_data, output_csv)

# Display extracted text as a DataFrame in Jupyter Notebook
df = pd.DataFrame(structured_data)
df.head()  # Display first few records
