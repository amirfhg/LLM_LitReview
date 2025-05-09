# Objective

This project aims to improve frontier models' performance to craft professional academic literature reviews. Our primary emphasis is on research papers within the social sciences, particularly in economics, finance, and MIS.

We first use fine-tuning as a warm-up supervised exercise. We then continue training the model under a reinforcement learning (RL) framework by creating a reward signal, guiding the model to perform a more effective literature review.  

# Data Collection

A curated training sample of target papers $$\prod_{train}$$ is collected from Semantic Scholar. These papers are carefully selected to ensure they are published in high-quality journals, such as QJE, JF, JFE, ISR. All papers in $$\prod_{train}$$ are published before 2020. We also collect a set of out-of-sample papers $$\prod_{test}$$ to evaluate model performance. These are the papers published after 2020. 

For each target paper $$p \in {\prod_{train}, \prod_{test}}$$, there is the set of papers referenced by $$p$$:

R<sub>p</sub> = {r<sub>p,1</sub>, r<sub>p,2</sub>, …, r<sub>p,N<sub>p</sub></sub>}

The papers in $$R_{p}$$ are important as they are used both during the fine-tuning and evaluation stages. 

For the fine-tuning stage, these papers together with the literature review of the target paper $$p$$ are used to train the model. During the fine-tuning, the model learns to write a literature review similar to the one in the target paper, $$p$$ based on the input papers in $$R_{p}$$. At this point, we do not use the full content of the referenced papers to fine-tune our model. Instead only rely on the metadata of the referenced papers. The metadata for papers in $$R_{p}$$ is retrieved from Semantic Scholar. This metadata includes the 'publication year,' 'authors' names,' 'title,' and 'abstract.' 

Similarly during the evaluation stage for each out-of-sample paper $$p$$, $$R_{p}$$ is given to the fine-tuned model and is asked to generate a literature review which is then evaluated against the one in $$p$$. 

The code to collect the metadata for target papers can be found in 'semantic_scholar_references.py' in this repository.

# Fine-tuning 

For each paper in $$\prod_{train}$$, training example consists of the following components:

1. An instruction prompt.
2. The target paper's literature review.
3. The target paper's research question.
4. Metadata of the target paper's references. 
The preparation of training data and its structure can be found in 'prepare_finetuning_dataset.py' in this repository.

# Evaluation Strategy

We must rely on a set of evaluation benchmarks to track the progress we make in improving foundation models' capabilities to generate academic literature reviews. 
To do so after each iteration of fine-tuning, we use our model to generate literature reviews for out-of-sample papers. We evaluate our model's output against those of foundation models (gpt4o) and actual professional academic literature reviews across multiple dimensions, including Coherence, Consistency, Fluency, and Relevance. 

***RQSim Benchmark***

We developed an evaluation benchmark, Research Question Similarity or RQSim, to gauge a model’s ability to identify gaps in academic literature and propose research questions that address those gaps. In our view, formulating the right research question is pivotal for writing a high-quality literature review. This is because a well-defined research question guides researchers in selecting and synthesizing relevant concepts and findings from the literature, ultimately shaping the structure and narrative of the review. Based on this notion, the core function of a literature review is to justify the paper's research question by presenting gaps in the literature. This means a model's ability to identify gaps and propose research questions is a predictor of its ability to synthesize quality literature reviews. Therefore, we measure the performance of our model based on its ability to identify gaps in the literature by reading papers in $$R_{p}$$ and propose a set of research questions similar to those suggested in the target paper $$p$$. 

For each paper $$p \in \prod_{test}$$, based on R<sub>p</sub> = {r<sub>p,1</sub>, r<sub>p,2</sub>, …, r<sub>p,N<sub>p</sub></sub>}, we instruct the fine-tuned model to generate a set of potential research questions:

Q<sub>p</sub> = {q<sub>p,1</sub>, q<sub>p,2</sub>, …, q<sub>p,n<sub>p</sub></sub>}  

The actual research question in paper $$p \in \prod_{test}$$ is $$RQ_{p}$$. We then use Sentence-Embedding (SBERT) to embedd $$RQ_{p}$$ and {q<sub>p,1</sub>, q<sub>p,2</sub>, …, q<sub>p,n<sub>p</sub></sub>}.

Next, we calculate $$RQSim(p)$$ to be the average cosine similarity between the vector embeddings of generated research questions in $$Q_{p}$$ and the vector embedding of $$RQ_{p}$$:

$$\[
RQSim(p) = \frac{1}{n_p} \sum_{i=1}^{n_p} \text{Cosine}(\vec{q_{p,i}}, \vec{RQ_{p}})
\]$$

As discussed higher values of $$RQSim(p)$$ indicates that model's generated research questions semantically converge to the ones proposed by professional academics in the same literature. The values of $$RQSim(p)$$ across the papers in $$\prod_{test}$$ reflect the model's performance in reading academic papers and suggesting research questions. Below are $$RQSim(p)$$ calculated for out-of-sample papers from $$\prod_{test}$$ across various frontier models, including our fine-tuned model based on gpt-4o.

<div align="center">
  <img src="https://github.com/user-attachments/assets/87a283e5-4e49-40b8-8b48-a58fe6388a55" alt="plot1" width="500">
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/3fbbaede-2941-47c8-aaef-c3c81f22308e" alt="boxplot" width="500">
</div>


# Reinforcement Learning 
RQSim metric can also be used as a reward signal for the model to evaluate its own performance. At each RL step we sample from a subset of $$\prod_{test}$$ (a different, much larger set of papers than the one used in fine-tuning stage), generate literature review using the fine-tuned model, calculate RQSim. Following a typical proximal policy optimization (PPO) method we calculate the following as the reward function at each step of RL:

$$\[
R = \alpha \cdot \text{RQSim} - \beta \cdot \text{IrrelevancePenalty}
\]$$

where $$\alpha$$ and $$\beta$$ are tunable hyperparameters. RQSim is the average of $$S(p)$$ over the selected papers. Irrelevance penalty is average of the following over the selected papers:

$$\[
\text{Penalty}(p) = \frac{1}{n_p} \sum_{i=1}^{n_p} 1 - \text{Cosine}(\vec{q_{p,i}}, \vec{Ref_{p}})
\]$$

where $$\vec{Ref_{p}}$$ is a single embedding of all the abstracts in R<sub>p</sub> = {r<sub>p,1</sub>, r<sub>p,2</sub>, …, r<sub>p,N<sub>p</sub></sub>}$. This ensures that the model does not deviate from the overall context of input papers. 

This section is not implemented yet. 







