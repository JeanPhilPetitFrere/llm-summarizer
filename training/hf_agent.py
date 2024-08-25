from optimum.quanto import QuantizedModelForCausalLM, qint4
from transformers import AutoModelForCausalLM, AutoTokenizer

class HuggingFaceAgent:
    def __init__(self, model_name) -> None:
        self.model = self.quantize_model(model_name)
        self.tokenizer = self.load_tokenizer(model_name)
    
    def load_model(self, model_name):
        return AutoModelForCausalLM.from_pretrained(model_name)
    
    def quantize_model(self,model_name):
        model = self.load_model(model_name)
        return QuantizedModelForCausalLM.quantize(model, weights=qint4, exclude='lm_head')
    
    def load_tokenizer(self, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer