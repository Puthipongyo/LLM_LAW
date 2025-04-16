from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd

# Load model & tokenizer
model_name = "openthaigpt/openthaigpt-1.0.0-70b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)


# ✅ สร้างฟังก์ชันสำหรับถามอะไรก็ได้
def ask_law_question(question):
    messages = [
        {"role": "user", "content": f"""ข้อมูลมีดังนี้:

        {context}

        คำถาม: {question}"""}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=400, do_sample=False)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("📌 คำตอบจากโมเดล:\n", response)


if __name__ == "__main__":
    # Load CSV
    context = ""
    df = pd.read_csv("Extracted_Law_CSV.csv", encoding="utf-8-sig")
    for i, row in df.head(3).iterrows():  
        context += f"[{i}] {row['text']}\n\n"
        
        
    ask_law_question("จากเนื้อเรื่องนี้ มีมาตราไหนเกี่ยวข้องบ้าง")
    ask_law_question("ข้อความไหนมีมาตรา ๒๒๖")
