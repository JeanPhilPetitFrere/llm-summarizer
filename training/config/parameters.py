import torch
from transformers import (
    TrainingArguments, 
    BitsAndBytesConfig, 
)
from peft import LoraConfig

# Type to cast the models submodules
compute_dtype = getattr(torch, "float16")

# Quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,  # loads the model with 4 bits precision Using QLoRA
    bnb_4bit_quant_type="nf4",  # Variant of 4bit configuration other being FP4 -> nf4 is more
    # resilient to temperature variation, better for llama series of model
    bnb_4bit_compute_dtype=compute_dtype,  # Type to cast the models submodules ->     # Submodules allow a module designer to split
    # a complex model into several pieces where all the submodules contribute to a single namespace,
    # which is defined by the module that includes the submodules
    bnb_4bit_use_double_quant=False,  # controls whether we use a second quantization to save an additional 0.4 bits per parameter
)

lora_alpha = (
    16,
)  # when optimizing with adam: tuning this is roughly the same as tuning the learning rate
# if the initialization was scaled properly

lora_dropout = (
    0.1,
)  # standard dropout: reduce overfitting by randomly selecting neurons to ignore with a dropout probability during training

r = (
    64,
)  # represents the rank of the low rank matrices learned during the finetuning process.
# As this value is increased, the number of parameters needed to be updated during the low-rank adaptation increases.
# a lower r may lead to a quicker, less computationally intensive training process, but may affect the quality of the model thus produced.
# However, increasing r beyond a certain value may not yield any discernible increase in quality of model output

bias = (
    "none",
)  # can be ‘none’, ‘all’ or ‘lora_only’. If ‘all’ or ‘lora_only’, the corresponding biases will be updated during training.
# Even when disabling the adapters, the model will not produce the same output as the base model would have without adaptation. The default is None

task_type = "CAUSAL_LM"

peft_params = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=r,
    bias=bias,
    task_type=task_type
)


# Set training parameters
training_params = TrainingArguments(
    output_dir="results",  # The output directory where the model predictions and checkpoints will be written.
    num_train_epochs=1,  # Total number of training epochs to perform.
    per_device_train_batch_size=4,  # The batch size per GPU/TPU core/CPU for training.
    gradient_accumulation_steps=1,  #  Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
    optim="paged_adamw_32bit",
    save_steps=100,  # Number of updates steps before two checkpoint saves.
    logging_steps=100,  # Number of update steps between two logs.
    learning_rate=2e-4,  # The initial learning rate for Adam.
    weight_decay=0.001,  # The weight decay to apply (if not zero).
    fp16=False,  # Whether to use 16-bit (mixed) precision training (through NVIDIA apex) instead of 32-bit training.
    bf16=False,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    max_grad_norm=0.3,  # Maximum gradient norm (for gradient clipping).
    max_steps=-1,  # If set to a positive number, the total number of training steps to perform. Overrides num_train_epochs.
    warmup_ratio=0.03,
    group_by_length=True,
    seed=42,
    lr_scheduler_type="cosine"  # "constant",
    # report_to="tensorboard"
)

