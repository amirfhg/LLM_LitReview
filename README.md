# Objective
This project aims to improve frontier models' performance to craft professional academic literature reviews. Our primary emphasis is on research papers within the social sciences, particularly in economics and finance.

# Data Collection
A curated training sample of target papers $$\prod_{train}$$ is collected from Semantic Scholar. These papers are carefully selected to ensure they are published in high-quality journals, such as QJE, JF, and JFE. All papers in $$\prod_{train}$$ are published before 2020.

We also collect a set of out-of-sample papers $$\prod_{test}$$ to evaluate model performance. These are the papers published after 2020. 

For each target paper $$p \in {\prod_{train}, \prod_{test}}$$, there is the set of referenced papers by $$p$$, R<sub>p</sub> = {r<sub>p,1</sub>, r<sub>p,2</sub>, …, r<sub>p,N<sub>p</sub></sub>}, $$N_{p}$$ being the number of papers referenced by $$p$$. 

The papers in $$R_{p}$$ are important as they are used both during the fine-tuning and evaluation stage. For the fine-tuning stage these papers along with the literature review of the target paper $$p$$ are used to train the model. During the fine-tuning the model learns on how to read a set of papers (referenced papers by $$p$$) and write a literature review similar to the actual one in $$p$$. 

At this point, we do not use the entirity of referenced papers to fine-tune our model. Instead only rely on the metadata of the referenced papers. The metadata for papers in $$R_{p}$$ is retrieved from Semantic Scholar. This metadata includes the 'publication year,' 'authors' names,' 'title,' and 'abstract.' 

Similarly during the evaluation stage for each out of sample paper $$p$$, $$R_{p}$$ is given to the fine-tuned model and is asked to generate a literature review which is then evaluated against the one in $$p$$. 

The code to collect the metadata for target papers can be found in 'semantic_scholar_references.py' in this repository.

# Fine-tuning 
For each paper in $$\prod_{train}$$, training example consists of the following components:

1. An instruction prompt.
2. The target paper's literature review.
3. The target paper's research question.
4. Metadata of the target paper's references.

The preparation of training data and the structure of each observation for fine-tuning can be found in 'prepare_finetuning_dataset.py' in this repository.

# Evaluation Strategy
We must develop and rely on a set of evaluation benchmarks to track the progress we make in improving foundation models' capabilities to generate academic literature reviews. 
To do so after each iteration of fine-tuning, we use our model to generate literature reviews for out-of-sample papers. We evaluate our model's output against those of foundation models (gpt4o) and actual professional academic literature reviews across multiple dimensions, including Coherence, Consistency, Fluency, and Relevance. 

We also develop our own evaluation strategy based on the notion that an ideal literature review should find gaps in the literature. Thus we expect our model to read a set of select papers given as input, identify gaps in the literature, and propose a set of research questions to expand on the current literature. Ultimately, our evaluation strategy aims to generate a set of research questions that converge to those proposed by the out-of-sample papers written by actual academics. 

$$p \in \prod_{test}$$ 

# Further Improvements
