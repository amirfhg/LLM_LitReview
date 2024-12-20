# -*- coding: utf-8 -*-
"""prepare_finetuning_dataset.ipynb

Here we collect all the necessary input data and prepare the dataset for finetuning. As described the three elements of fine-tuning dataset are:

1. The target paper's literature review.
2. The target paper's research question.
3. Metadata of the target paper's references.
4. An instruction prompt.

1. Extract Intro/LitReview from Target Papers (benchmark litreview)
"""

import os
import fitz  
import re

target_papers = [file for file in os.listdir('./') if file.endswith('.pdf')]


def clean_extracted_text(text):
    """
    Cleans extracted text by removing unwanted characters and formatting issues.
    """
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(r'\b(cid:[0-9]+)\b', '', text)  # Remove unwanted placeholders like (cid:1234)
    text = re.sub(r'\[.*?\]', '', text)  # Remove content within brackets []
    return text.strip()

def remove_unwanted_content(text):
    """
    Removes tables, figures, footnotes, and page numbers from the text using regex patterns.
    """
    text = re.sub(r'Table\s+\d+:.*?(\n|$)', '', text, flags=re.IGNORECASE)  # Remove table captions
    text = re.sub(r'Figure\s+\d+:.*?(\n|$)', '', text, flags=re.IGNORECASE)  # Remove figure captions
    text = re.sub(r'(Table|Figure)\s+\d+.*?(\n|$)', '', text, flags=re.IGNORECASE)  # Remove generic table/figure mentions
    text = re.sub(r'\d+\s*-\s*\d+', '', text)  # Remove page numbers (e.g., "1 - 2")
    text = re.sub(r'\[[^\]]*\]', '', text)  # Remove footnotes enclosed in brackets
    return text.strip()

def extract_text_from_pages(file_path):
    """
    Extracts and processes text from the first 10 pages of a PDF, removing tables, figures, footnotes, and page numbers.
    """
    try:
        doc = fitz.open(file_path)
        aggregated_text = ""

        for page_number in range(1, min(10, len(doc))):  # Process only the first 10 pages (exclude title page)
            page = doc[page_number]
            page_text = page.get_text("text")
            page_text = remove_unwanted_content(page_text)  # Remove tables, figures, footnotes, page numbers
            aggregated_text += page_text.strip() + " "  # Aggregate text from all pages

        doc.close()
        return clean_extracted_text(aggregated_text)  # Clean and return the aggregated text
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return ""

intros = []
for pdf in target_papers[0:1]:
    intro_text = extract_text_from_pages(pdf)
    intros.append(intro_text)
    

"""2. Extract Research Question from Target Papers"""

# Here we use gpt-4o to extract the research question of each target paper

from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
llm = ChatOpenAI(model="gpt-4o", api_key="?", temperature = 0)

research_q_list = []

template = """Use the following pieces of context to answer the question at the end.
        Do not give information not mentioned in the context information.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"], template=template
)

# Create the chain using LLMChain
chain = LLMChain(prompt=prompt, llm=llm, output_parser=StrOutputParser())

query = "What is the main research question discussed in the context? Formulate your response in a research question form."

# Iterate through elements of intros
for intro in intros:
    context_str = str(intro)
    # Use the 'predict' method for multiple input variables
    research_q = chain.predict(context=context_str, question=query)
    research_q_list.append(research_q)

"""3. Metadata for Referenced Papers by Target Paper"""

# Here we load the metadata collected from Semantic Scholar into a single string for each target paper

import pandas as pd
target_papers_references = [paper.replace('.pdf', '.csv') for paper in target_papers]
metadata_list = []

for reference in target_papers_references:
    # Load the CSV file
    df = pd.read_csv(reference)

    # Replace NaN in 'publicationYear' with '?'
    df['publicationYear'] = df['publicationYear'].fillna('?')

    # Remove '[' and ']' from 'authors'
    df['authors'] = df['authors'].str.replace(r'\[', '', regex=True).str.replace(r'\]', '', regex=True)

    # Create the metadata string for the current file
    metadata_string = '||'.join(
        f"title:'{row['title']}'. abstract:'{row['abstract']}'. authors:'{row['authors']}'. pubyear:'{row['publicationYear']}'"
        for _, row in df.iterrows()
    )

    # Append the metadata string to the list
    metadata_list.append(metadata_string)

"""4. Define Instruction Prompt and Save the whole Dataset as .json"""

import openai
import json

data = []

# Iterate through your observations to construct the dataset
for i in range(len(intros)):
    instruction_prompt = f"""The following is a list of paper metadata separated by ||.
     Each element in the list includes: title, abstract, author names, publication year.
     The items in this list are the papers referenced by the target paper. list of paper metadata = {metadata_list[i]}.
     The following is the research question from the target paper. research question = '{research_q_list[i]}'.
     Using abstract of papers content in the list of paper metadata, and considering the research question, learn to write a the target paper's literature review.
     Remember target paper's literature review may contain material that are not directly or indirectly related to the content in the list of paper metadata. Ignore those parts in target paper's literature review.
     The following is target paper's literature review:"""
    target_paper_litreview = intros[i]
    data.append({"prompt": instruction_prompt, "completion": completion})


# Step 2: Save dataset to a JSONL file (required format for fine-tuning)
fine_tune_file = "./fine_tune_data.jsonl"
with open(fine_tune_file, 'w') as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

print(f"Fine-tuning dataset saved to {fine_tune_file}")