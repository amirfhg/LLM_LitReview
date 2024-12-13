from semanticscholar import SemanticScholar
import requests
import json
import pandas as pd
import time
import os

# This code finds tile and abstract and other metadata of 'cited papers' within each target paper in train_sample 
# The results are saved as a .csv with the same name as target paper .pdf (which is target paper's corpus id in semantic scholar) 
# The same code is then implemented for papers in test_sample as well

# Mount directory 
wdir = r'./train_sample'
os.chdir(wdir)

# List all target papers' PDF files in the directory 
# Note. target papers are handpicked from semantic scholar and saved by their corpus id. This part must be done manually.
# Target papers are handpicked to ensure collecting quality papers for fine-tuning.
# We use corpus id to retrive target paper's references and their metadata from semantic schoalr

target_papers = [file for file in os.listdir('./') if file.endswith('.pdf')]
corpus_ids = [paper.split('/')[-1].replace('.pdf', '') for paper in target_papers]

# Initialize SemanticScholar instance
sch = SemanticScholar()

# Loop through each CorpusId and process data
for i, corpus_id in enumerate(corpus_ids, start=1):
    try:
        # Get paper details using Semantic Scholar API
        results = sch.get_paper(f'CorpusId:{corpus_id}')
        paper_url = results.url
        paper_id = paper_url.split('/')[-1]

        # Make API request to fetch references
        r = requests.post(
            'https://api.semanticscholar.org/graph/v1/paper/batch',
            params={'fields': 'referenceCount,references,title'},
            json={"ids": [paper_id]}
        )
        r.raise_for_status()  # Ensure the request was successful

        # Extract paper IDs from references
        paper_ids = [
            reference['paperId'] for paper in r.json()
            for reference in paper.get('references', [])
            if reference.get('paperId') is not None
        ]

        # Initialize an empty list to hold extracted data
        all_data = []

        # Loop through paper IDs and fetch title and abstract
        for index, pid in enumerate(paper_ids, start=1):
            try:
                r_ = requests.post(
                    'https://api.semanticscholar.org/graph/v1/paper/batch',
                    params={'fields': 'publicationDate,authors,title,abstract'},
                    json={"ids": [pid]}
                )
                r_.raise_for_status()  # Ensure the request was successful

                # Extract data from the response
                extracted_data = [
                    {
                        'publicationYear': paper['publicationDate'].split('-')[0] if paper.get('publicationDate') else None,
                        'authors': [author['name'] for author in paper.get('authors', []) if author.get('name')],
                        'title': paper.get('title', None),
                        'abstract': paper.get('abstract', None)
                    }
                    for paper in r_.json()
                ]

                # Append to the list
                all_data.extend(extracted_data)

                print(f"Processing reference {index} out of {len(paper_ids)}")

            except Exception as data_error:
                print(f"Error extracting data for paper ID {pid} at index {index}: {data_error}")
                continue

            # Delay to avoid exceeding request thresholds
            time.sleep(1)

        # Create a DataFrame from the collected data
        df = pd.DataFrame(all_data)
        df_cleaned = df.dropna(subset=['abstract'])

        # Save cleaned data to a CSV file
        df_cleaned.to_csv(f'{corpus_id}.csv', index=False)
        print(f"Data for CorpusId {corpus_id} saved successfully.")

    except Exception as e:
        print(f"Error processing CorpusId {corpus_id}: {e}")
        continue
