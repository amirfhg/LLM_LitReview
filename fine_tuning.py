# -*- coding: utf-8 -*-
import openai

# Fine-tune the GPT-4o model using the prepared JSONL file
# Set your OpenAI API key
openai.api_key = "?"

# Path to the JSONL file you created in 'prepare_finetuning_dataset.py'
fine_tune_file = "./fine_tune_data.jsonl"

# 1. Upload the file to OpenAI
file_response = openai.File.create(
    file=open(fine_tune_file, "rb"),
    purpose='fine-tune'
)
file_id = file_response["id"]

# Create the fine-tuning job
fine_tuning_job = openai.FineTuningJob.create(
    training_file=file_id,
    model="gpt-4o"
)

print(fine_tuning_job)
