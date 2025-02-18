# Import necessary libraries
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging
from peft import LoraConfig
from trl import SFTTrainer
import torch
import gc
from dGClip import *
#Force garbage collection
gc.collect()


# Install required libraries (uncomment the following line when running in a notebook environment)

#For PyTorch memory management add the following code

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"



# Define model, dataset, and new model name
base_model = "NousResearch/Llama-2-7b-chat-hf"
guanaco_dataset = "mlabonne/guanaco-llama2-1k"
new_model = "llama-2-7b-chat-guanaco"


class CustomTrainer(SFTTrainer):
    def create_optimizer(self):
        """Overrides the default optimizer creation with DeltaGclip."""
        print("Model parameters ",self.model.parameters())
        self.optimizer = dGClip(self.model.parameters(), lr=0.01)
        return self.optimizer

# Load dataset
dataset = load_dataset(guanaco_dataset, split="train")

# 4-bit Quantization Configuration
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=False)

# Load model with 4-bit precision
model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant_config, device_map={"": 0})
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Set PEFT Parameters
peft_params = LoraConfig(lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM")

# Define training parameters
training_params = TrainingArguments(output_dir="./results", num_train_epochs=1, per_device_train_batch_size=4, gradient_accumulation_steps=1, save_steps=25, logging_steps=25, learning_rate=2e-4, weight_decay=0.001, fp16=False, bf16=False, max_grad_norm=0.3, max_steps=-1, warmup_ratio=0.03, group_by_length=True, lr_scheduler_type="constant", report_to="tensorboard")

# Initialize the trainer
trainer = CustomTrainer(model=model, train_dataset=dataset, peft_config=peft_params, dataset_text_field="text", max_seq_length=None, tokenizer=tokenizer, args=training_params, packing=False)

#Force clean the pytorch cache
gc.collect()

torch.cuda.empty_cache()

# Train the model
trainer.train()

# Save the model and tokenizer
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

# Evaluate the model (optional, requires Tensorboard installation)
# from tensorboard import notebook
# log_dir = "results/runs"
# notebook.start("--logdir {} --port 4000".format(log_dir))

# Test the model
logging.set_verbosity(logging.CRITICAL)
prompt = "Who is Leonardo Da Vinci?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

