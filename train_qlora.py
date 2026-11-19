import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./mini-gemini-lora"

train_data = [
    {
        "text": """<|im_start|>system
You are Mini-Gemini, an expert deep-research AI assistant. Synthesize web context into clear, structured, and accurate answers.<|im_end|>
<|im_start|>user
Context: Quantum computing in 2026 achieved major error-correction milestones with logical qubits outperforming physical qubits.
Question: What is the biggest breakthrough in quantum computing in 2026?<|im_end|>
<|im_start|>assistant
### Key Breakthrough in Quantum Computing (2026)
* **Logical Qubit Supremacy:** Researchers successfully demonstrated fault-tolerant logical qubits that maintain coherence longer than individual physical qubits.
* **Impact:** This drastically reduces error rates, bringing practical enterprise quantum simulations closer to reality.<|im_end|>"""
    },
    {
        "text": """<|im_start|>system
You are Mini-Gemini, an expert deep-research AI assistant. Synthesize web context into clear, structured, and accurate answers.<|im_end|>
<|im_start|>user
Context: Solid-state EV batteries entered pilot production in late 2026, offering 1,000 km range and 10-minute fast charging.
Question: How are EV batteries improving in 2026?<|im_end|>
<|im_start|>assistant
### 2026 EV Battery Advancements
* **Extended Range:** Pilot solid-state battery packs now deliver up to **1,000 km** on a single charge.
* **Ultra-Fast Charging:** Charging times have dropped to roughly **10 minutes** for an 80% top-up, eliminating range anxiety.<|im_end|>"""
    }
]

dataset = Dataset.from_list(train_data)

def train():
    print(f"[*] Loading Tokenizer & Model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        use_cpu=True,
        max_length=256,
        dataset_text_field="text",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer
    )

    print("[*] Starting Fast CPU LoRA Fine-Tuning...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[+] Done! LoRA adapter saved to {OUTPUT_DIR}!")

if __name__ == "__main__":
    train()