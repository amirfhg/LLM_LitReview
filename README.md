## Automate_LitReview
This is a finetuning project on OpenAI's foundation models to improve their capabilities in writing professional academic literature review. Our primary focus is on papers from social sciences, mainly economics and finance. 

# Data Collection
A training sample of target papers are collected from Semantic Scholar. These papers are handpicked to ensure the quality of journals they are published in (e.g. QJE, JF, etc.).

For each target paper, the metadata for references are retrieved from Semantic Scholar as well. This metadata includes 'publication year', 'authors names', 'title', 'abstract'. The literature review in the target papers are the benchmark. Both the references' metadata (title+ abstract) and target papers' literature review are used as finetuning dataset. 

Each training datapoint includes: [instruction prompt + target paper's literature review + target paper's research question + metadata of target paper's references]
Hundreds of training examples are then used to fine tune gpt-4o. 
