import time
import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from config.settings import (
    LLM_MODEL_PATH,
    MAX_LLM_TOKENS
)

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_dtype = torch.float16 if device == "cuda" else torch.float32


tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL_PATH
)


llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_PATH,
    device_map={"": device},
    dtype=compute_dtype
)


def clean_answer(text):

    text = text.split("\n")[0]

    text = text.replace("*", "")
    text = text.replace("**", "")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()



def generate_answer(question):


    if not question or not str(question).strip():

        return {
            "answer":"متوجه سوال نشدم",
            "latency":0,
            "tokens":0
        }



    prompt = f"""
شما یک سامانه پرسش و پاسخ فارسی هستید.

قوانین:
- فقط فارسی جواب بده.
- کوتاه جواب بده.
-اگر سوال نظر شخصی داشت، با لحن مناسب پاسخ بدهید.
-اگر پاسخ را نمی‌دانی صادقانه بگو.

سوال:
{question}
"""


    messages = [
        {
            "role":"user",
            "content":prompt
        }
    ]


    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(
        llm_model.device
    )


    start = time.time()


    with torch.inference_mode():

        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=MAX_LLM_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )


    latency = time.time()-start


    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )


    answer = clean_answer(
        generated
    )


    tokens = len(
        tokenizer.encode(
            answer,
            add_special_tokens=False
        )
    )


    return {
        "answer":answer,
        "latency":latency,
        "tokens":tokens
    }
