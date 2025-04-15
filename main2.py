import fitz
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Load text from PDF
doc = fitz.open("/workspace/test/law_1page.pdf")
page_text = doc[0].get_text()
doc.close()

# 2. Load model & tokenizer
model_name = "scb10x/typhoon2-qwen2.5-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 3. Prepare prompt
prompt = f"""ในข้อความต่อไปนี้ ให้ช่วยหาว่าคำว่า "มาตรา ๘" ปรากฏอยู่หรือไม่ พร้อมระบุบริบทโดยรอบ:\n\n{page_text}"""

# 4. Tokenize with truncation
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

# 5. Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 6. Generate output
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
