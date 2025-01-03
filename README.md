# Objective

This project aims to improve frontier models' performance to craft professional academic literature reviews. Our primary emphasis is on research papers within the social sciences, particularly in economics and finance.

We first use fine-tuning as a warm-up supervised exercise. We then continue training the model under a reinforcement learning (RL) framework by creating a reward signal, guiding the model to perform more effective literature review.  

# Data Collection

A curated training sample of target papers $$\prod_{train}$$ is collected from Semantic Scholar. These papers are carefully selected to ensure they are published in high-quality journals, such as QJE, JF, and JFE. All papers in $$\prod_{train}$$ are published before 2020. We also collect a set of out-of-sample papers $$\prod_{test}$$ to evaluate model performance. These are the papers published after 2020. 

For each target paper $$p \in {\prod_{train}, \prod_{test}}$$, there is the set of referenced papers by $$p$$:

R<sub>p</sub> = {r<sub>p,1</sub>, r<sub>p,2</sub>, …, r<sub>p,N<sub>p</sub></sub>}

The papers in $$R_{p}$$ are important as they are used both during the fine-tuning and evaluation stage. 

For the fine-tuning stage these papers along with the literature review of the target paper $$p$$ are used to train the model. During the fine-tuning the model learns on how to read a set of papers (referenced papers by $$p$$) and write a literature review similar to the actual one in $$p$$. At this point, we do not use the entirity of referenced papers to fine-tune our model. Instead only rely on the metadata of the referenced papers. The metadata for papers in $$R_{p}$$ is retrieved from Semantic Scholar. This metadata includes the 'publication year,' 'authors' names,' 'title,' and 'abstract.' 

Similarly during the evaluation stage for each out of sample paper $$p$$, $$R_{p}$$ is given to the fine-tuned model and is asked to generate a literature review which is then evaluated against the one in $$p$$. 

The code to collect the metadata for target papers can be found in 'semantic_scholar_references.py' in this repository.

# Fine-tuning 

For each paper in $$\prod_{train}$$, training example consists of the following components:

1. An instruction prompt.
2. The target paper's literature review.
3. The target paper's research question.
4. Metadata of the target paper's references.

Below is the example of each entry for fine-tuning: 

> The following is a list of paper metadata separated by '|'.  
> Each element in the list includes: title, abstract, author names, publication year.  
> The items in this list are the papers referenced by the target paper.  
> **list of paper metadata** = `{metadata_list[i]}`.  
> The following is the research question from the target paper.  
> **research question** = `'{research_q_list[i]}'`.  
> Using abstract of papers content in the list of paper metadata, and considering the research question, learn to write the target paper's literature review.  
> Remember target paper's literature review may contain material that is not directly or indirectly related to the content in the list of paper metadata.  
> Ignore those parts in the target paper's literature review.  
> The following is the target paper's literature review:  
> **target_paper_litreview** = `intros[i]`

    


The preparation of training data and its strucure can be found in 'prepare_finetuning_dataset.py' in this repository.

# Evaluation Strategy

We must rely on a set of evaluation benchmarks to track the progress we make in improving foundation models' capabilities to generate academic literature reviews. 
To do so after each iteration of fine-tuning, we use our model to generate literature reviews for out-of-sample papers. We evaluate our model's output against those of foundation models (gpt4o) and actual professional academic literature reviews across multiple dimensions, including Coherence, Consistency, Fluency, and Relevance. 

***RQSim Benchmark***

We also developed our benchmark, RQSim, based on the notion that performing a literature review should ideally lead to identifying gaps in the literature and suggest research questions to address them. Thus we expect our model to identify gaps in the literature by reading out-of-sample papers, and propose a set of research questions comparable to those suggested by professional academics. 

For each paper $$p \in \prod_{test}$$, based on R<sub>p</sub> = {r<sub>p,1</sub>, r<sub>p,2</sub>, …, r<sub>p,N<sub>p</sub></sub>}, we instruct the fine-tuned model to generate a set of potetial research questions:

Q<sub>p</sub> = {q<sub>p,1</sub>, q<sub>p,2</sub>, …, q<sub>p,n<sub>p</sub></sub>}  

The actual research question in paper $$p \in \prod_{test}$$ is $$RQ_{p}$$. We then use Sentence-Embedding (SBERT) to embedd $$RQ_{p}$$ and {q<sub>p,1</sub>, q<sub>p,2</sub>, …, q<sub>p,n<sub>p</sub></sub>}.

Next, we calculate the average cosine similarity between the vector embeddings of generated research questions in $$Q_{p}$$ and the vector embedding of $$RQ_{p}$$:

$$\[
S(p) = \frac{1}{n_p} \sum_{i=1}^{n_p} \text{Cosine}(\vec{q_{p,i}}, \vec{RQ_{p}})
\]$$

As discussed higher values of $$S(p)$$ indicates that model's generated research questions semantically converge to the ones proposed by professional academics in the same literature. The average values of $$S(p)$$ across the papers in $$\prod_{test}$$ is RQSim. This metric is used to compare the performance of the fine-tuned model in each iteration with its past iterations and other models (e.g. NotebookLM, o1, gpt-4o). 

This section is not implemented yet. 

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







