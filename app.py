import streamlit as st
import os
from training.trainer import LLMTrainer
import logging
from training.config.variables import training_dataset_location
from config.text import (
    page_title,
    step_1_instruction
    
)
from transformers import AutoModelForCausalLM

st.set_page_config(layout="wide", page_title=page_title)

st.markdown(
    f'<p style="color:black;font-family:DMS-sans;font-size:70px;border-radius:2%;">{page_title}</p>',
    unsafe_allow_html=True,
)

if not os.path.isfile("------- TBD model path --------"):
    logging.info("Training", extra={"step":"Trainer Init"})
    llm_trainer = LLMTrainer()
    logging.info("Training", extra={"step":"Training start"})
    llm_trainer.train(training_dataset_location,"article","summary")
    logging.info("Training", extra="Training end")

st.markdown(
    f'<p style="color:black;font-family:DMS-sans;font-size:30px;border-radius:2%;">{step_1_instruction}</p>',
    unsafe_allow_html=True,
)

st.write("test")