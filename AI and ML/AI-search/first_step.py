import json
import numpy as np
import unicodedata
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Model and tokenizer initialization
model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')

def normalize(text):
    """Normalize unicode text (NFC) and strip whitespace."""
    return unicodedata.normalize("NFC", text.strip())

def encode_text(text):
    """
    Encode text into an embedding, truncating to 512 tokens.
    Returns a normalized embedding.
    """
    encoded = tokenizer.encode(text, truncation=True, max_length=512)
    truncated_text = tokenizer.decode(encoded, skip_special_tokens=True)
    return model.encode(truncated_text, normalize_embeddings=True)

def process_chunk(chunk):
    """
    Creates embeddings for the text and its metadata features.
    Combines several embedding types for enhanced semantic retrieval.
    """
    text_embedding = encode_text(chunk['text'])
    key_phrases_embedding = encode_text(' '.join(chunk.get('key_phrases', [])))
    lemmas_embedding = encode_text(' '.join(chunk.get('lemmas', [])))

    entities_text = ' '.join([entity['text'] for entity in chunk.get('entities', [])])
    entities_embedding = encode_text(entities_text) if entities_text else np.zeros_like(text_embedding)

    # Weighted combination of feature embeddings
    combined_embedding = (
        0.5 * text_embedding +
        0.3 * key_phrases_embedding +
        0.1 * lemmas_embedding +
        0.1 * entities_embedding
    )

    return {
        'text': chunk['text'],
        'text_embedding': text_embedding.tolist(),
        'key_phrases_embedding': key_phrases_embedding.tolist(),
        'lemmas_embedding': lemmas_embedding.tolist(),
        'entities_embedding': entities_embedding.tolist(),
        'combined_embedding': combined_embedding.tolist()
    }

def prepare_data(json_file):
    """
    Loads a JSON file and creates embeddings for each chunk,
    grouping results by document (filename and path).
    """
    grouped_documents = defaultdict(lambda: {"file_name": "", "file_path": "", "chunks": []})

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

        for entry in data:
            file_name = normalize(entry['file_name'])
            file_path = entry['file_path']  # do not normalize path
            file_key = (file_name, file_path)

            grouped_documents[file_key]["file_name"] = file_name
            grouped_documents[file_key]["file_path"] = file_path

            for chunk in entry['chunks']:
                embeddings = process_chunk(chunk)
                grouped_documents[file_key]["chunks"].append(embeddings)

    return list(grouped_documents.values())

def save_embeddings(documents, output_file):
    """Saves embeddings and metadata to output as a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # INPUT provides path to source JSON, OUTPUT is where the results will be saved.
    input_json = "/path/to/all_files_result.json"
    output_json = "grouped_document_embeddings.json"

    documents = prepare_data(input_json)
    save_embeddings(documents, output_json)

    print("Embeddings saved to", output_json)


