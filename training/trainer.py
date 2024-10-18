from training.dataset import DatasetGenerator
from trl import SFTTrainer
from training.hf_agent import HuggingFaceAgent
from training.config.variables import (
    base_model,
    save_dir,
)
from training.config.parameters import (
    training_params,
    peft_params,
)
from training.config.processing import get_target_modules
import traceback

class LLMTrainer:
    """
    LLM Fine-tuning class

    ...

    Attributes
    ----------
    base_model : str
        name of the base model in HF

    save_dir : str
        The saving location for the fine-tuned model

    quantization_config : BitsAndBytesConfig
        parameters of the quantization

    training_parameters : TrainingArguments
        parameters of the LLM fine-tuning


    Methods
    -------
    generate_prompt(input,output):
        Format the data into the proper format for LLM training

    get_dataset(file_path,response_col,candidate_label_col,assigned_labels_col):
        Creates a dataset for fine-tuning from a csv file

    train(training_data_path,response_col,candidate_label_col,assigned_labels_col):
        fine-tunes a LLM and saves it locally

    """

    def __init__(
        self,
        base_model: str = base_model,
        save_dir: str = save_dir,
        training_parameters=training_params,
    ) -> None:
        """
        Constructs all the necessary attributes for the training

        ...

        Parameters
        ----------
        base_model : str
            name of the base model in HF

        save_dir : str
            The saving location for the fine-tuned model

        quantization_config : BitsAndBytesConfig
            parameters of the quantization

        training_parameters : TrainingArguments
            parameters of the LLM fine-tuning
        """
        self.hf_agent = HuggingFaceAgent(base_model)
        # Load model
        self.model = self.hf_agent.model

        # Load tokenizer
        self.tokenizer = self.hf_agent.tokenizer

        # PEFT arguments
        self.peft_params = peft_params
        self.peft_params.target_modules = get_target_modules(self.model)

        # training parameters
        self.training_parameters = training_parameters

        # Save directory
        self.save_dir = save_dir

        # Dataset generator
        self.get_dataset = DatasetGenerator().get_dataset
    

    def train(self, training_data_path: str, input_col: str, output_col: str):
        """
        fine-tunes a LLM and saves it locally

        Parameters
        ----------
        training_data_path : str
            path to the JSON

        input_col : str
            column in the csv that stores the inputs

        output_col : str
            column in the csv that stores the summary outputs

        Returns
        -------
        A saved fine-tuned model in the selected local directory
        """
        # Set supervised fine-tuning parameters
        dataset = self.get_dataset(training_data_path, input_col, output_col)

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            args=self.training_parameters,
            peft_config=self.peft_params,
            tokenizer=self.tokenizer,
        )

        try:
            trainer.train()
            # Save trained model
            trainer.model.save_pretrained(self.save_dir)
            trainer.tokenizer.save_pretrained(self.save_dir)
            return "training successful"

        except Exception as e:
            print(f"An error has occured: {e}")
            traceback.print_exc()
