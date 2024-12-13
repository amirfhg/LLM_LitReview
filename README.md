# Automate_LitReview
This project focuses on fine-tuning OpenAI's foundation models to enhance their ability to craft professional academic literature reviews. Our primary emphasis is on research papers within the social sciences, particularly in economics and finance.

# Data Collection
A curated training sample of target papers is collected from Semantic Scholar. These papers are carefully selected to ensure they are published in high-quality journals, such as QJE, JF, and JFE.

For each target paper, metadata for its references is also retrieved from Semantic Scholar. This metadata includes the 'publication year,' 'authors' names,' 'title,' and 'abstract.' The literature review sections of the target papers serve as the benchmark. Both the references' metadata (title and abstract) and the target papers' literature reviews are incorporated into the fine-tuning dataset.

Each training example consists of the following components:

An instruction prompt.
The target paper's literature review.
The target paper's research question.
The metadata of the target paper's references.
Hundreds of such training examples are compiled and used to fine-tune GPT-4/4o.

# Fine-tuning 
# Generated LitReview Eval
