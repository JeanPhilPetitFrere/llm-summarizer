import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
import re
def merge_and_save_model(save_dir:str,base_model:str):
    """
    Will merge the fine-tuned weights to the original model and save it to the chosen dir

    Parameters
    ----------
    save_dir : str
        fine-tuned model save directory
    
    base_model : str
        HF identifier of the base model

    Returns
    -------
    A saved & merged fine-tuned model in the selected local directory
    """
    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
                base_model,
                low_cpu_mem_usage=True,
                return_dict=True,
                torch_dtype=torch.float16,
                device_map={"": 0},
            )
    try:
        model = PeftModel.from_pretrained(base_model, save_dir)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(save_dir, safe_serialization=True)
        return "Merge done"
    except Exception as e:
        return f"An unexpected error as occured: {e}"

def get_target_modules(model: AutoModelForCausalLM) -> list:
    """
    Search the llm architecture for all of the linear layers, so that we can update them during fine-tuning

    Parameters
    ----------
    model : AutoModelForCausalLM
        The LLM

    Returns
    -------
    target_modules : list
        list of all of the linear layers in the LLM architecture
    """
    model_modules = str(model.modules)
    pattern = r"\((\w+)\): Linear"
    linear_layer_names = re.findall(pattern, model_modules)

    names = []
    # Print the names of the Linear layers
    for name in linear_layer_names:
        names.append(name)
    target_modules = list(set(names))
    return target_modules