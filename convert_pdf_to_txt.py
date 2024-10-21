import pdfplumber
import os

# Đường dẫn tới file PDF và tệp .txt
pdf_path = "data/STSV-2024-ONLINE.pdf"
txt_output_path = os.path.join(os.path.dirname(pdf_path), "cleaned_STSV-2024.txt")

## Hàm đọc PDF và lưu nội dung vào tệp .txt
def extract_and_save_pdf_as_txt(pdf_path, txt_output_path):
    cleaned_content = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw_text = page.extract_text()
            if raw_text:
                # Loại bỏ xuống dòng không cần thiết
                cleaned_content += raw_text.replace('\n', ' ').lower().strip() + "\n\n"

    # Lưu nội dung đã làm sạch với mã hóa UTF-8
    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_content)

# Gọi hàm để đọc PDF và lưu thành tệp .txt
extract_and_save_pdf_as_txt(pdf_path, txt_output_path)
print(f"PDF đã được lưu thành tệp: {txt_output_path}")

