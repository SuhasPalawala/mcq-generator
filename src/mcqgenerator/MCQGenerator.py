import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

#importing necessary packages from langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain


load_dotenv()

KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(api_key=KEY, model_name = "gpt-4o", temperature = 0.3)

TEMPLATE = """
Text: {text}
You are an expert MCQ test maker. Given the above text, it is your job to \n
create a quiz of {number} multiple choice questions for {subject} students in a {tone} tone.
Make sure the questions are not repeated and check to make sure that all the questions can be solved with information \n
from the text. Make sure to format your questions like the RESPONSE_JSON below.
Ensure that you make {number} questions.

RESPONSE_JSON: {response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables = ["text", "number", "subject", "tone", "response_json"],
    template = TEMPLATE
)

quiz_chain = LLMChain(llm = llm, prompt = quiz_generation_prompt, output_key = "quiz", verbose = True)

TEMPLATE2 = """
You are an English writer who is very proficient in the English langauge, especically grammar. Given a \n
multiple choice quiz for {subject} students, you need to evaluate the complexity of the questions and provide a \n
complexity analysis of the quiz. Only use at max 50 words for the ocmplexity analysis. If the quiz is not on par with \n
the cognitive and analytical abilities of the students, update the quiz questions which need to be changed so that the tone \n
is appropriate for the students' cogntive level.
Quiz MCQs:
{quiz}

Check from an expert English writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables = ["subject", "quiz"],
    template = TEMPLATE2
)

review_chain = LLMChain(llm = llm, prompt = quiz_evaluation_prompt, output_key = "review", verbose = True)

generate_evaluate_chain = SequentialChain(
    chains = [quiz_chain, review_chain],
    input_variables = ["text", "number", "subject", "tone", "response_json"],
    output_variables = ["quiz", "review"],
    verbose = True,
)

