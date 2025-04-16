import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()


model_name = "openthaigpt/openthaigpt-1.0.0-70b-chat"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16
)


def ask_law_question(question, context):
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

    print("📌 คำถาม:", question)
    print("📌 คำตอบจากโมเดล:\n", response)
    print("=" * 80)


if __name__ == "__main__":

    df = pd.read_csv("Extracted_Law_CSV.csv", encoding="utf-8-sig")


    context = ""
    for i, row in df.head(3).iterrows():
        context += f"[{i}] {row['text']}\n\n"


    ask_law_question("จากเนื้อเรื่องนี้ มีมาตราไหนเกี่ยวข้องบ้าง", context)
    ask_law_question("ข้อความไหนมีมาตรา ๒๒๖", context)
