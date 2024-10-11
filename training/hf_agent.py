from optimum.quanto import QuantizedModelForCausalLM, qint4
from transformers import AutoModelForCausalLM, AutoTokenizer


class HuggingFaceAgent:
    """
    Handles the interactions with HugginFace

    Attributes
    ----------
    model : QuantizedModelForCausalLM
        Quantized LLM

    tokenizer : AutoTokenizer
        Tokenizer for the LLM

    Methods
    -------
    _load_model(model_name)
        loads a model from the Huggingface hub

    quantize_model(model_name)
        Applies quantization to the selected LLM

    load_tokenizer(tokenizer_name)
        loads the corresponding tokenizer to the model
    """

    def __init__(self, model_name: str) -> None:
        """
        Parameters
        ----------
        model_name : str
            Name of the model in the HuggingFaceHub
        """
        # TODO: Check __getattribute__ vs __getattr__
        self.model = self.quantize_model(model_name)
        self.tokenizer = self.load_tokenizer(model_name)

    def _load_model(self, model_name: str):
        """
        loads a model from the Huggingface hub

        Parameters
        ----------
        model_name : str
            Name of the model in the HuggingFaceHub

        Returns
        -------
        AutoModelForCausalLM
            loaded model
        """
        return AutoModelForCausalLM.from_pretrained(model_name)

    def quantize_model(self, model_name: str):
        """
        Applies quantization on the loaded model

        Parameters
        ----------
        model_name : str
            Name of the model in the HuggingFaceHub

        Returns
        -------
        QuantizedModelForCausalLM
            Quantized model
        """
        model = self._load_model(model_name)
        return QuantizedModelForCausalLM.quantize(
            model, weights=qint4, exclude="lm_head"
        )

    def load_tokenizer(self, tokenizer_name: str):
        """
        Loads the corresponding tokenizer

        Parameters
        ----------
        tokenizer_name : str
            Name of the tokenizer in the HuggingFaceHub

        Returns
        -------
        AutoTokenizer
            Loaded tokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer
