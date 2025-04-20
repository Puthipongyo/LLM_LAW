import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import HfFolder

# ตั้งค่าป้องกัน CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# 1. Load model & tokenizer
model_name = "scb10x/llama3.1-typhoon2-70b-instruct"
token = HfFolder.get_token()  # ใช้ token ที่ login ไว้ในเครื่องนี้

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
    token=token
)

# 2. ฟังก์ชันถามคำถามจาก context
def ask_law_question(question, context, return_text=False):
    messages = [
        {"role": "system", "content": "คุณคือนักกฎหมายผู้เชี่ยวชาญกฏหมายแพ่งและพาณิชย์ หน้าที่ของคุณคือรับสถานการณ์จาก user prompt และตอบเฉพาะเลขมาตรากฎหมายแพ่งและพาณิชย์ 5 อันดับที่เกี่ยวข้องที่สุดเท่านั้น"},
        {"role": "user", "content": f"บริบท:\n{context}\n\nคำถาม:\n{question}"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )
    response = outputs[0][input_ids.shape[-1]:]
    decoded = tokenizer.decode(response, skip_special_tokens=True)

    return decoded.strip()


# 3. Main process
if __name__ == "__main__":
    df = pd.read_csv("/workspace/test/data_case_100.csv", encoding="utf-8-sig")

    if "text" not in df.columns:
        raise ValueError("ไม่พบ column 'text' ในไฟล์ CSV")

    # เตรียมคอลัมน์ผลลัพธ์
    df['predicted_law'] = ""

    # วนลูปรันแต่ละแถว
    for i in range(df.shape[0]):
        question = df['text'].iloc[i]

        # context อาจใช้จากก่อนหน้า หรือเฉพาะแถวนี้ก็ได้
        context = f"{question}"

        full_prompt = f"""
        {question}

        ข้อความคดีนี้มีมาตราแพ่งและพาณิชย์ที่เกี่ยวข้องมากที่สุด 5 อันดับแรก มีอะไรบ้าง (บอกแค่มาตรา)
        """.strip()

        ans = ask_law_question(full_prompt, context)
        df.at[i, 'predicted_law'] = ans
        print(f"[{i}] ✅ Answer: {ans}")

        if i == 2:  # ทดสอบแค่ 3 แถวแรก
            break

    # บันทึกผล
    df.head(3).to_csv("/workspace/test/output_predicted.csv", index=False, encoding="utf-8-sig")
