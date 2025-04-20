import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import HfFolder
import gc

# ตั้งค่าป้องกัน CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# โหลด model & tokenizer
model_name = "scb10x/llama3.1-typhoon2-70b-instruct"
token = HfFolder.get_token()

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

# ฟังก์ชันถามคำถามจาก context
def ask_law_question(question, context):
    torch.cuda.empty_cache()
    gc.collect()
    messages = [
        {"role": "system", "content": "คุณคือนักกฎหมายผู้เชี่ยวชาญกฎหมายแพ่งและพาณิชย์ หน้าที่ของคุณคือรับสถานการณ์จาก user prompt และตอบเฉพาะเลขมาตรากฎหมายแพ่งและพาณิชย์ 5 อันดับที่เกี่ยวข้องที่สุดเท่านั้น"},
        {"role": "user", "content": f"""
        บริบท:\n{context}\n\nคำถาม:\n{question}
        """},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,  # ลดขนาด output
        eos_token_id=[tokenizer.eos_token_id],
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )

    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True).strip()

# ฟังก์ชันวัด F1 แบบเซต
def f1_set(pred, gold):
    try:
        pred_set = set(map(int, pred.split(',')))
        gold_set = set(map(int, str(gold).split(',')))
    except:
        return 0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(gold_set) if gold_set else 0
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

# MAIN PROCESS
if __name__ == "__main__":
    df = pd.read_csv("/workspace/test/data_case_100.csv", encoding="utf-8-sig")

    if "text" not in df.columns or "answers" not in df.columns:
        raise ValueError("ไม่พบ column 'text' หรือ 'answers' ในไฟล์ CSV")

    df['predicted_law'] = ""
    df['f1'] = 0.0

    # for i in range(df.shape[0]):  # ✅ รันครบทุกแถว
    #     question = df['text'].iloc[i]
    #     context = question

    #     prompt = f"""
    #     {question}
    #     ข้อความคดีนี้มีมาตราแพ่งและพาณิชย์ที่เกี่ยวข้องมากที่สุด 5 อันดับแรก มีอะไรบ้าง (ตอบแค่เลขมาตรา เช่น 515,234)
    #     """.strip()

        # pred = ask_law_question(prompt, context)
        # df.at[i, 'predicted_law'] = pred

        # gold = df['answers'].iloc[i]
        # f1 = f1_set(pred, gold)
        # df.at[i, 'f1'] = f1

        # print(f"[{i}] ✅ Predicted: {pred} | Answer: {gold} | F1 Score: {f1:.2f}")
        
    #     if i == 20 :
    #         break

    i = 60
    question = df['text'].iloc[i]
    context = question
    prompt = f"""   
    #     {question}
    #     ข้อความคดีนี้มีมาตราแพ่งและพาณิชย์ที่เกี่ยวข้องมากที่สุด 5 อันดับแรก มีอะไรบ้าง (ตอบแค่เลขมาตรา เช่น 515,234)
    #     """.strip()
    pred = ask_law_question(prompt, context)
    df.at[i, 'predicted_law'] = pred

    gold = df['answers'].iloc[i]
    f1 = f1_set(pred, gold)
    df.at[i, 'f1'] = f1

    print(f"[{i}] ✅ Predicted: {pred} | Answer: {gold} | F1 Score: {f1:.2f}")

    # ✅ บันทึกผลลัพธ์
    df.to_csv("/workspace/test/output_predicted.csv", index=False, encoding="utf-8-sig")
