import os
import torch
import fitz  # PyMuPDF
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ตั้งค่าป้องกัน CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# 1. Load text from PDF
doc = fitz.open("/workspace/test/law_1page.pdf")
page_text = doc[0].get_text()
doc.close()

# 2. Load model & tokenizer (multi-GPU + 4bit)
model_name = "scb10x/llama3.1-typhoon2-70b"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",  # ✅ ใช้ทุก GPU อัตโนมัติ
    torch_dtype=torch.float16
)

# 3. Prepare prompt
prompt = f"""จากข้อความใน PDF ต่อไปนี้:

\"\"\"{page_text}\"\"\"

คำถาม: ข้อความที่ให้ไปในมาตรา ๖ มีมาตราอะไรมาเกี่ยวข้องด้วย

ตอบ:
"""

# 4. Tokenize & send to GPU
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# 5. Generate answer
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=400, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)


# 7. Print the final result
print("\n=== คำตอบจากโมเดล ===\n")
print(response)
