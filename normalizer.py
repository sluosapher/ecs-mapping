import csv
#import spacy
#import numpy as np
import json
from annoy import AnnoyIndex
import os
from myembedder import MyEmbedder, DIMENSION_OF_EMBEDDINGS

# Load spacy's pre-trained English model
# nlp = spacy.load('en_core_web_md')

# Read text content from a local csv file
def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)  # Read CSV file with headers
        data = [row for row in reader]  # Convert csv.DictReader object to list of dicts
    return data

# Compute embedding of each line
def compute_embeddings(embedder, data, fields):

    # prepare the texts for embedding
    texts = []
    for index, line in enumerate(data):
        text = ' '.join(line[field] for field in fields)
        texts.append(text)
    
    # compute the embeddings
    embeddings = embedder.get_embeddings_batch(texts)
    

    # prepare the embeddings for saving
    records = []
    for index in range(len(texts)):
        records.append({"chunk_id": index, "text": texts[index], "embedding": embeddings[index]}) 

    return records

# Save embeddings to a json file
def save_embeddings(embeddings, file_path):
    with open(file_path, 'w') as file:
        json.dump(embeddings, file)

# Load embeddings from a json file and build an Annoy index
def build_annoy_index(embedding_file_path, dims=DIMENSION_OF_EMBEDDINGS):
    t = AnnoyIndex(dims, 'angular')
    with open(embedding_file_path, 'r') as file:
        embeddings = json.load(file)
        for emb in embeddings:
            t.add_item(emb['chunk_id'], emb['embedding'])
    t.build(10)  # 10 trees
    return t, embeddings

# Main program
def main():

    embeddings_file_path = 'embeddings.json'
    embedder = MyEmbedder()

    # If embeddings.json exists, load it. Otherwise, compute the embeddings and save them to embeddings.json
    if os.path.exists(embeddings_file_path):
        with open(embeddings_file_path, 'r') as file:
            embeddings = json.load(file)
        t, _ = build_annoy_index(embeddings_file_path)
    else:
        # Step 1: Read text content from a local csv file
        data = read_csv('ECS fields.csv')

        # Fields to keep for each line of text
        fields = ["Field_Set", "Field", "Level", "Description"]

        # Step 2: Compute embedding of each line and save the embeddings
        embeddings = compute_embeddings(embedder, data, fields)
        save_embeddings(embeddings, 'embeddings.json')

        # Step 3: Load all embeddings with the Annoy library and create an index
        t, _ = build_annoy_index('embeddings.json')

    while True:
        # Step 4: Ask user to input a line of query text
        query_text = input("Enter a line of query text or 'exit' to quit: ")

        if len(query_text) == 0:
            continue
        
        # Break the loop if the user wants to exit
        if query_text.lower() == 'exit':
            break

        # Step 5: Compute the embedding of the query text with the same spacy model
        query_vector = embedder.get_embeddings([query_text])[0]

        # Step 6: Perform similarity search with Annoy for the query text
        idxs, dists = t.get_nns_by_vector(query_vector, 3, include_distances=True)

        # Step 7: Return the top 3 texts that are semantically closest to the query text
        # print the distances for each result, and order the results by distance
        for idx, dist in sorted(zip(idxs, dists), key=lambda x: x[1]):
            print(f"{embeddings[idx]['text']} - {dist}")

        # for idx in idxs:
        #     print(embeddings[idx]['text'])

        print('\n\n')

if __name__ == '__main__':
    main()
