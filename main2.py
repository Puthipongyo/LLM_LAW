import fitz  # PyMuPDF
import requests


doc = fitz.open("law_1page.pdf")
page_text = doc[0].get_text()
doc.close()


prompt = f"สรุปความหมายของ มาตรา ๗:\n\n{page_text}"


API_URL = "http://116.109.210.48:46504/generate"

payload = {
    "inputs": prompt,
    "parameters": {
        "max_new_tokens": 200,
        "temperature": 0.7
    }
}

res = requests.post(API_URL, json=payload)
print(res.json())
