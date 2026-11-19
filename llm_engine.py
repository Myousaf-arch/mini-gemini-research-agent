import os
from threading import Thread
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer
)
from peft import PeftModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

LOCAL_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
LORA_ADAPTER_DIR = "./mini-gemini-lora"

# Non-chat / audio / guard models ko skip karne ke liye
SKIP_MODEL_KEYWORDS = ("whisper", "guard", "tts", "playai", "embed", "moderation")
FALLBACK_CLOUD_MODELS = ("llama-3.3-70b-versatile", "llama-3.1-8b-instant")


class MiniGeminiEngine:
    def __init__(
        self,
        model_name: str = LOCAL_MODEL_NAME,
        lora_dir: str = LORA_ADAPTER_DIR,
        use_lora: bool = False,
        groq_api_key: str = None
    ):
        self.api_key = (groq_api_key or os.getenv("GROQ_API_KEY", "")).strip()
        self.use_cloud = False
        self.cloud_model = None
        self.chat_models = []
        self.client = None
        self.local_model = None
        self.tokenizer = None

        self.system_prompt = (
            "You are Mini-Gemini, an elite, high-precision AI research analyst. "
            "Synthesize the provided web search context into a clear, authoritative, well-structured answer.\n\n"
            "STRICT RULES:\n"
            "1. NEVER repeat the same fact multiple times or list raw search engine labels.\n"
            "2. Lead immediately with a direct, comprehensive answer in the first paragraph.\n"
            "3. Use clean Markdown headings (###), concise bullet points, and **bold key terms**.\n"
            "4. Cross-verify facts across sources; ignore irrelevant/spam snippets completely.\n"
            "5. If the context does not fully answer a specific part, combine verified general knowledge clearly while prioritizing the retrieved sources."
        )

        if self.api_key and GROQ_AVAILABLE:
            try:
                self.client = Groq(api_key=self.api_key)
                available = self.client.models.list().data
                self.chat_models = [
                    m.id for m in available
                    if not any(bad in m.id.lower() for bad in SKIP_MODEL_KEYWORDS)
                ]
                print(f"[*] Groq Available Models: {self.chat_models}")

                if self.chat_models:
                    preferred = [
                        m for m in self.chat_models
                        if any(k in m.lower() for k in ("70b", "32b", "scout", "maverick", "llama", "qwen", "gemma", "gpt"))
                    ]
                    self.cloud_model = preferred[0] if preferred else self.chat_models[0]
                    self.use_cloud = True
                    print(f"[*] Selected Ultra-Fast Cloud Model: {self.cloud_model}")
            except Exception as e:
                print(f"[!] Groq discovery warning: {e}")

        if not self.use_cloud:
            self._init_local_model(model_name, lora_dir, use_lora)

    def _init_local_model(self, model_name: str, lora_dir: str, use_lora: bool):
        if self.local_model is not None and self.tokenizer is not None:
            return
        print(f"[*] Initializing Local CPU Engine ({model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )

        if use_lora and os.path.exists(lora_dir):
            self.local_model = PeftModel.from_pretrained(base_model, lora_dir)
        else:
            self.local_model = base_model

        self.local_model.eval()

    def _format_context(self, retrieved_docs: list) -> str:
        if not retrieved_docs:
            return "No external web sources retrieved."
        blocks = []
        for idx, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("source", "Web")
            title = doc.metadata.get("title", "Source")
            content = doc.page_content[:1200].strip()
            blocks.append(f"[Source {idx}: {title} | {source}]\n{content}")
        return "\n\n---\n\n".join(blocks)

    def _stream_local(self, user_payload: str, max_new_tokens: int = 350):
        self._init_local_model(LOCAL_MODEL_NAME, LORA_ADAPTER_DIR, use_lora=False)
        hf_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_payload}
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            hf_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.local_model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            repetition_penalty=1.18,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )

        thread = Thread(target=self.local_model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    def stream_answer(
        self,
        question: str,
        retrieved_docs: list,
        chat_history: str = "",
        continue_from: str = ""
    ):
        formatted_context = self._format_context(retrieved_docs)

        if continue_from:
            tail_excerpt = continue_from[-1200:]
            user_payload = (
                f"Original Question: {question}\n\n"
                f"Verified Web Context:\n{formatted_context}\n\n"
                f"Answer Written So Far (ending at):\n{tail_excerpt}\n\n"
                "Continue seamlessly from the exact point where the previous answer stopped. "
                "Do not repeat already-written sections; finish any cut-off sentence/point and complete all remaining headings/phases cleanly."
            )
        else:
            user_payload = (
                f"Verified Web Context:\n{formatted_context}\n\n"
                f"Recent Conversation:\n{chat_history[-1000:]}\n\n"
                f"User Question: {question}"
            )

        if self.use_cloud and self.client:
            candidate_models = [self.cloud_model] + [
                m for m in FALLBACK_CLOUD_MODELS if m in self.chat_models or not self.chat_models
            ]
            unique_models = list(dict.fromkeys([m for m in candidate_models if m]))

            for idx, model_name in enumerate(unique_models):
                try:
                    stream = self.client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_payload}
                        ],
                        temperature=0.3,
                        max_tokens=960,
                        top_p=0.9,
                        stream=True
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            yield delta
                    return
                except Exception as e:
                    print(f"[!] Groq model '{model_name}' failed/rate-limited: {e}")
                    if idx < len(unique_models) - 1:
                        print(f"[*] Falling back to '{unique_models[idx + 1]}'...")
                        continue

        yield from self._stream_local(user_payload)