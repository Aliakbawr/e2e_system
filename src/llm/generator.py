import time
import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from config.settings import (
    LLM_MODEL_PATH,
    MAX_LLM_TOKENS,
    MAX_LLM_INPUT_TOKENS,
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



SYSTEM_INSTRUCTIONS = """شما یک دستیار گفتگومحور فارسی هستید.

قوانین:
- فقط فارسی جواب بده.
- کوتاه و مرتبط جواب بده.
- برای فهم ارجاع‌ها و پرسش‌های ادامه‌دار از تاریخچه گفتگو استفاده کن.
- اگر کاربر نظر شخصی خواست، با لحن مناسب پاسخ بده.
- اگر سؤال مبهم است یا پاسخ را نمی‌دانی، صادقانه بگو.
- در متن گفتاری از حروف لاتین استفاده نکن.
- اصطلاحات انگلیسی را در صورت وجود با معادل رایج فارسی بنویس.
- نام‌ها و اصطلاحات بدون معادل را با تلفظ فارسی بنویس.
- مثال: Python ← پایتون، Linux ← لینوکس، CPU ← سی‌پی‌یو.
- اگر نام یا اصطلاح انگلیسی در پاسخ لازم است، نسخه گفتاری آن را با خط فارسی بنویس.
- از حروف لاتین استفاده نکن.
مثال نادرست:
Answer:
از بهترین مدل‌های جی‌پی‌یو می‌توان به GPT-3، LaMDA و BERT اشاره کرد.
مثال درست:
Answer:
از بهترین مدل‌های جی‌پی‌یو می‌توان به جی‌پی‌تی-۳، لامدا و برت اشاره کرد.
"""


def _build_messages(question, history=None, memory=None, turn_context=None):
    """Build a Gemma-compatible alternating conversation."""
    messages = [dict(message) for message in (history or [])]
    messages.append({"role": "user", "content": str(question).strip()})

    # Gemma 2's standard template expects alternating user/model roles and may
    # reject a separate system role. Put the instructions in the first user
    # message without modifying the session's stored copy.
    memory_block = str(memory or "").strip()
    prefix_parts = [SYSTEM_INSTRUCTIONS]
    if memory_block:
        prefix_parts.append(memory_block)
    prefix_parts.append(f"پیام کاربر:\n{messages[0]['content']}")
    messages[0]["content"] = "\n\n".join(prefix_parts)

    # Keep confirmed memory close to the current question as well. With a full
    # recent-history window, placing it only in the oldest user message makes a
    # small local model prone to following newer but unrelated turns instead.
    current_parts = []
    if memory_block and len(messages) > 1:
        current_parts.append(f"یادآوری حافظه مرتبط:\n{memory_block}")
    if turn_context:
        current_parts.append(str(turn_context).strip())
    if current_parts:
        current_parts.append(f"پیام فعلی کاربر:\n{messages[-1]['content']}")
        messages[-1]["content"] = "\n\n".join(current_parts)
    return messages


def _render_with_recent_history(
    question,
    history=None,
    memory=None,
    turn_context=None,
):
    """Render a prompt, dropping oldest complete turns if it is too large."""
    retained_history = list(history or [])

    # History consists of complete user/assistant pairs. Removing two entries
    # maintains the role alternation required by Gemma's chat template.
    while True:
        messages = _build_messages(
            question,
            retained_history,
            memory,
            turn_context,
        )
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_count = len(tokenizer.encode(text, add_special_tokens=False))

        if token_count <= MAX_LLM_INPUT_TOKENS or len(retained_history) < 2:
            return text

        retained_history = retained_history[2:]


def generate_answer(question, history=None, memory=None, turn_context=None):


    if not question or not str(question).strip():

        return {
            "answer":"متوجه سوال نشدم",
            "latency":0,
            "tokens":0
        }

    text = _render_with_recent_history(question, history, memory, turn_context)


    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LLM_INPUT_TOKENS
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
