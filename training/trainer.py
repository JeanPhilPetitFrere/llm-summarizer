import json
import re
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, DatasetDict
from peft import LoraConfig
from trl import SFTTrainer
from sklearn.model_selection import train_test_split
from fine_tuning_configs.general_config import base_model, save_dir,system_prompt
from fine_tuning_configs.quantization_config import quant_config
from fine_tuning_configs.training_config import training_params




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
        quantization_config=quant_config,
        training_parameters=training_params,
        system_prompt: str=system_prompt
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

        ####### System prompt
        self.system_prompt=system_prompt
       
        ####### Load model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map={"": 0}
            # device_map="auto"
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        self.model = model

        ####### Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer

        ####### Get target modules
        model_modules = str(model.modules)
        pattern = r"\((\w+)\): Linear"
        linear_layer_names = re.findall(pattern, model_modules)

        names = []
        # Print the names of the Linear layers
        for name in linear_layer_names:
            names.append(name)
        target_modules = list(set(names))

        ####### PEFT arguments
        self.peft_args = LoraConfig(
            lora_alpha=16,  # when optimizing with adam: tuning this is roughly the same as tuning the learning rate
            # if the initialization was scaled properly
            lora_dropout=0.1,  # standard dropout: reduce overfitting by randomly selecting neurons to ignore with a dropout probability during training
            r=64,  # represents the rank of the low rank matrices learned during the finetuning process.
            # As this value is increased, the number of parameters needed to be updated during the low-rank adaptation increases.
            # a lower r may lead to a quicker, less computationally intensive training process, but may affect the quality of the model thus produced.
            # However, increasing r beyond a certain value may not yield any discernible increase in quality of model output
            bias="none",  # can be ‘none’, ‘all’ or ‘lora_only’. If ‘all’ or ‘lora_only’, the corresponding biases will be updated during training.
            # Even when disabling the adapters, the model will not produce the same output as the base model would have without adaptation. The default is None
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

        ####### Other variables

        # training parameters
        self.training_parameters = training_parameters
        # Save directory
        self.save_dir = save_dir

    def generate_prompt(
        self,
        text_input: dict,
        expected_output: str = None,
        training : bool = False
    ) -> str:
        """Format the text for LLM training

        Parameters
        ----------
        sentence : str
            Sentence to be classified

        candidate_label_dict : str
            string dictionary of the labels and their description

        expected_output : str
            expected assigned labels from the candidate label list
        
        training : bool
            wheter the prompt is to be used for training or not

        Returns
        -------
        prompt : str
            Formatted sentence to be used for LLM fine-tuning
        """
        formated_input = {
            "text": text_input
        }
        if training:
            prompt = f"""[INST] <SYS> {self.system_prompt} </SYS>
            Input:\n###
            {formated_input}
            ###
            Output:[/INST] {expected_output}
                """.strip()
        else:
            prompt = f"""[INST] <SYS> {self.system_prompt} </SYS>
            Input:\n###
            {formated_input}
            ###
            Output:[/INST]""".strip()
         
        return prompt
        

    def get_dataset(
        self,
        file_path: str,
        input_col: str,
        output_col: str
    ) -> DatasetDict:
        """
        Creates a dataset for fine-tuning from a json file for open-ends classification

        Parameters
        ----------
        file_path : str
            path to the csv

        input_col : str
            column in the csv that stores the inputs

        output_col : str
            column in the csv that stores the summary outputs

        Returns
        -------
        dataset : DatasetDict
            Formatted dataset, ready to be used for llm fine-tuning
        """
        # Data loading
        df = pd.read_csv(file_path,index_col=0)


        df["training_prompt"] = df.apply(
            lambda row: self.generate_prompt(
                row[input_col], row[output_col], True
            ),
            axis=1,
        )
        df["testing_prompt"] = df.apply(
            lambda row: self.generate_prompt(
                row[input_col]
            ),
            axis=1,
        )

        # train test eval split
        train_ratio = 0.75
        validation_ratio = 0.15
        test_ratio = 0.10

        # train is now 75% of the entire data set
        train, test = train_test_split(df, test_size=1 - train_ratio)

        # test is now 10% of the initial data set
        # validation is now 15% of the initial data set
        eval, test = train_test_split(test, test_size=test_ratio/(test_ratio + validation_ratio))

        train_dataset = Dataset.from_pandas(train)
        eval_dataset = Dataset.from_pandas(eval)
        test_dataset = Dataset.from_pandas(test)

        # storing it in a HF dataset
        dataset = DatasetDict()
        dataset["train"] = train_dataset
        dataset["eval"] = eval_dataset
        dataset["test"] = test_dataset
        return dataset

    def train(
        self,
        training_data_path: str,
        input_col: str,
        output_col: str
    ):
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
        dataset = self.get_dataset(
            training_data_path,
            input_col,
            output_col
        )
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            peft_config=self.peft_args,
            dataset_text_field="training_prompt",
            max_seq_length=None,
            tokenizer=self.tokenizer,
            args=self.training_parameters,
            packing=False,
        )
        try:
            trainer.train()
            # Save trained model
            trainer.model.save_pretrained(self.save_dir)
            trainer.tokenizer.save_pretrained(self.save_dir)
            return "training successful"

        except Exception as e:
            return f"An unexpected error as occured: {e}"
