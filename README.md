# Objective
This project aims to improve frontier models' performance to craft professional academic literature reviews. Our primary emphasis is on research papers within the social sciences, particularly in economics and finance.

# Data Collection
A curated training sample of target papers $$\prod_{train}$$ is collected from Semantic Scholar. These papers are carefully selected to ensure they are published in high-quality journals, such as QJE, JF, and JFE.

For each target paper $$p \in \prod_{train}$$, there is the set of referenced papers by $$p$$, R<sub>p</sub> = {r<sub>p,1</sub>, r<sub>p,2</sub>, …, r<sub>p,N<sub>p</sub></sub>}, $$N_{p}$$ being the number of papers referenced by $$p$$.

The metadata for references is retrieved from Semantic Scholar. This metadata includes the 'publication year,' 'authors' names,' 'title,' and 'abstract.' The literature review sections of the target papers serve as the benchmark. Both the references' metadata (title and abstract) and the target papers' literature reviews are incorporated into the fine-tuning dataset.

# Fine-tuning 
$$\prod_{train}$$ is the set of papers used for fine-tuning. All of the papers in $$\prod_{train}$$ are published before 2020. For each paper in $$\prod_{train}$$, training example consists of the following components:

1. An instruction prompt.
2. The target paper's literature review.
3. The target paper's research question.
4. Metadata of the target paper's references.

# Evaluation Strategy
We use another set of papers $$\prod_{train}$$, published after 2020 to test our model's performance. We must develop and rely on a set of evaluation benchmarks to track the progress we make in improving foundation models' capabilities to generate academic literature reviews. Accordingly after each iteration of fine-tuning, we use our model to generate literature reviews for out-of-sample papers. These papers are published after the papers in the training sample but belong to the same journals. 

We evaluate our model's output against those of foundation models (gpt4o) and actual professional academic literature reviews across multiple dimensions, including Coherence, Consistency, Fluency, and Relevance. 

We also develop our evaluation strategy based on the notion that an ideal literature review should find gaps in the literature. Thus we expect our model to read a set of select papers given as input, identify gaps in the literature, and propose a set of research questions to expand on the current literature. Ultimately, our evaluation strategy aims to generate a set of research questions that converge to those proposed by the out-of-sample papers written by actual academics. 

$$p \in \prod_{test}$$ 

# Further Improvements
