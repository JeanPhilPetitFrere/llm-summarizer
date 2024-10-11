from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd
from training.config.variables import system_prompt


class DatasetGenerator:
    def __init__(self):
        ####### System prompt
        self.system_prompt = system_prompt

    def generate_prompt(
        self, text_input: dict, expected_output: str = None, training: bool = False
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
        formated_input = {"text": text_input}
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
        self, file_path: str, input_col: str, output_col: str
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
        df = pd.read_csv(file_path, index_col=0)

        df["training_prompt"] = df.apply(
            lambda row: self.generate_prompt(row[input_col], row[output_col], True),
            axis=1,
        )
        df["testing_prompt"] = df.apply(
            lambda row: self.generate_prompt(row[input_col]),
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
        eval, test = train_test_split(
            test, test_size=test_ratio / (test_ratio + validation_ratio)
        )

        train_dataset = Dataset.from_pandas(train)
        eval_dataset = Dataset.from_pandas(eval)
        test_dataset = Dataset.from_pandas(test)

        # storing it in a HF dataset
        dataset = DatasetDict()
        dataset["train"] = train_dataset
        dataset["eval"] = eval_dataset
        dataset["test"] = test_dataset
        return dataset
