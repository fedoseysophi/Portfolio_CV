import json
import numpy as np
import os

def fill_missing_combined_embeddings(input_filename, output_filename):
    # Check if input file exists
    if not os.path.exists(input_filename):
        raise FileNotFoundError(f"File not found: {input_filename}")

    # Load list of documents
    with open(input_filename, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    # Fill missing or empty 'combined_embedding' field for each document
    for doc in documents:
        # If field is missing or empty
        if not doc.get('combined_embedding'):
            # Collect chunk embeddings
            chunk_embs = [
                np.array(chunk['combined_embedding'])
                for chunk in doc.get('chunks', [])
                if chunk.get('combined_embedding')
            ]
            if chunk_embs:
                # Calculate mean and convert to Python list
                avg_emb = np.mean(chunk_embs, axis=0)
                doc['combined_embedding'] = avg_emb.tolist()
            else:
                # If no chunk embeddings are present
                doc['combined_embedding'] = []

    # Save updated documents to output file
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(documents)} documents. Result saved to '{output_filename}'.")

if __name__ == "__main__":
    INPUT_FILE = "grouped_document_embeddings.json"
    OUTPUT_FILE = "grouped_document_embeddings_filled.json"
    fill_missing_combined_embeddings(INPUT_FILE, OUTPUT_FILE)

