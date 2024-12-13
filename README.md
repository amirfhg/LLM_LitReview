# Automate_LitReview
This project focuses on fine-tuning OpenAI's foundation models to enhance their ability to craft professional academic literature reviews. Our primary emphasis is on research papers within the social sciences, particularly in economics and finance.

# Data Collection
A training sample of target papers are collected from Semantic Scholar. These papers are handpicked to ensure the quality of journals they are published in (e.g. QJE, JF, etc.).

For each target paper, the metadata for references are retrieved from Semantic Scholar as well. This metadata includes 'publication year', 'authors names', 'title', 'abstract'. The literature review in the target papers are the benchmark. Both the references' metadata (title+ abstract) and target papers' literature review are used as parts of the finetuning dataset. 

Each training datapoint includes: [instruction prompt + target paper's literature review + target paper's research question + metadata of target paper's references]
Hundreds of training examples are then used to fine tune gpt-4o. 

# Fine-tuning 
# Generated LitReview Eval
