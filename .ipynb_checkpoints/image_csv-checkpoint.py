import pytesseract
from PIL import Image
import csv
import pandas as pd

# Path to your image file
image_path = "AminulSirIMG.jpeg"
output_csv = "image_csv.csv"

# Convert image to text using Tesseract
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="ben+eng")  # Specify both Bengali (ben) and English (eng)
    return text

# Save extracted text to CSV
def save_text_to_csv(text, output_csv):
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Split text by line and write each line to CSV
        for line in text.splitlines():
            writer.writerow([line])

# Main processing
text = extract_text_from_image(image_path)
save_text_to_csv(text, output_csv)

# Display extracted text
print("Extracted Text:\n")
print(text)

# Load CSV into Pandas DataFrame for better visualization
df = pd.read_csv(output_csv, header=None, names=["Extracted Text"])
df.head()  # Display first few lines
