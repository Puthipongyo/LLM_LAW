import os
import torch
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional: for memory safety
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# 1. Load text from PDF
doc = fitz.open("/workspace/test/law_1page.pdf")
page_text = doc[0].get_text()
doc.close()

# 2. Load model & tokenizer
model_name = "scb10x/typhoon2-qwen2.5-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 3. Prepare question
prompt = f"""จากข้อความใน PDF ต่อไปนี้:

\"\"\"{page_text}\"\"\"

คำถาม: มาตรา ๑๑ ในข้อความหมายถึงอะไร? ช่วยอธิบายบริบทให้เข้าใจง่าย

ตอบ:
"""

# 4. Tokenize & prepare input
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

# 5. Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 6. Generate output
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 7. Print the full response
print("\n=== คำตอบจากโมเดล ===\n")
print(response)
