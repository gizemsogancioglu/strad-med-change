from pathlib import Path

from huggingface_hub import file_exists

from source.data_processing.data_reader import data_path, clinical_texts
from source.data_processing.data_model import ClinicalText, Trajectory

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import time

def get_embeddings(text, tokenizer, model, max_length=512, device='cuda'):
    """
    Return averaged BERT embedding for a document, handling long text (>512 tokens)
    and ensuring input tensors are always 2D.
    """
    # Tokenize without truncation
    encoded = tokenizer(text, return_tensors='pt', truncation=False)
    input_ids = encoded['input_ids'][0]        # 1D tensor
    attention_mask = encoded['attention_mask'][0]

    embeddings = []

    for i in range(0, input_ids.size(0), max_length):
        chunk_ids = input_ids[i:i + max_length].unsqueeze(0).to(device)   # 2D
        chunk_mask = attention_mask[i:i + max_length].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_ids=chunk_ids, attention_mask=chunk_mask)

        # Average pooling over sequence dimension
        emb = outputs.last_hidden_state.mean(dim=1)
        embeddings.append(emb)

    # Average across all chunks
    doc_embedding = torch.vstack(embeddings).mean(dim=0)
    return doc_embedding.squeeze().cpu().numpy()


def get_avg_bert_features(texts, device='cuda', pretrained_model_path=None):
    model_name = pretrained_model_path or "CLTL/MedRoBERTa.nl"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    return [get_embeddings(t, tokenizer, model, device=device) for t in texts]


def save_bert_features(text_data, results_path, device='cuda', pretrained_model_path=None):
    start_time = time.time()
    # Extract embeddings on GPU
    bert_features = get_avg_bert_features(text_data[ClinicalText.TEXT.value].tolist(), device=device,
                                         pretrained_model_path=pretrained_model_path)

    # Create dataframe
    bert_cols = [f'bert_feature_{i}' for i in range(768)]
    bert_df = pd.DataFrame(bert_features, columns=bert_cols)

    # Add trajectory_id
    bert_df[Trajectory.ID.value] = text_data[Trajectory.ID.value].values
    bert_df[ClinicalText.NOTE_ID.value] = text_data[ClinicalText.NOTE_ID.value].values
    # Save as Parquet
    output_file = "bert_features.parquet" if pretrained_model_path is None else "bert_features_finetuned.parquet"
    bert_df.to_parquet(results_path/output_file, engine="pyarrow")

    print(f"Saved {len(bert_df)} rows to {output_file}")
    print(f"Time: {time.time() - start_time:.2f} seconds")

    return bert_df
    start_time = time.time()

    # Extract embeddings on GPU
    bert_features = get_avg_bert_features(text_data[ClinicalText.TEXT.value].tolist(), device=device)

    # Create dataframe
    bert_cols = [f'bert_feature_{i}' for i in range(768)]
    bert_df = pd.DataFrame(bert_features, columns=bert_cols)

    # Add trajectory_id
    bert_df[Trajectory.ID.value] = text_data[Trajectory.ID.value].values
    bert_df[ClinicalText.NOTE_ID.value] = text_data[ClinicalText.NOTE_ID.value].values
    # Save as Parquet
    output_file = "bert_features.parquet"
    bert_df.to_parquet(results_path/output_file, engine="pyarrow")

    print(f"Saved {len(bert_df)} rows to {output_file}")
    print(f"Time: {time.time() - start_time:.2f} seconds")

    return bert_df


def save_tfidf_features(text_data, results_path, max_features=500):
    """
    Compute TF-IDF features for text data, save to Parquet, and return dataframe.

    Args:
        text_data (pd.DataFrame): DataFrame containing text and trajectory/note IDs.
        max_features (int): Maximum number of TF-IDF features.

    Returns:
        pd.DataFrame: TF-IDF features with trajectory_id and note_id.
    """
    start_time = time.time()

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(text_data[ClinicalText.TEXT.value].tolist())

    # Convert sparse matrix to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_feature_{i}" for i in range(tfidf_matrix.shape[1])])

    # Add trajectory_id and note_id
    tfidf_df[Trajectory.ID.value] = text_data[Trajectory.ID.value].values
    tfidf_df[ClinicalText.NOTE_ID.value] = text_data[ClinicalText.NOTE_ID.value].values

    # Save to Parquet
    output_file = f"tfidf_features_{max_features}.parquet"
    tfidf_df.to_parquet(results_path/output_file, engine="pyarrow")

    print(f"Saved {len(tfidf_df)} rows to {output_file}")
    print(f"Time: {time.time() - start_time:.2f} seconds")

    return tfidf_df

def read_bert_features(finetuned=False):
    filename = 'bert_features_finetuned.parquet' if finetuned else 'bert_features.parquet'
    file_path = data_path / filename

    data = pd.read_parquet(file_path).drop_duplicates()

    return data

def read_tfidf_features(dim=None):
    if dim is None:
        filename = 'tfidf_features.parquet'
    else:
        filename = f'tfidf_features_{dim}.parquet'
    file_path = data_path / filename

    data = pd.read_parquet(file_path).drop_duplicates()

    return data


if __name__ == "__main__":

    bert_file_path = data_path / "bert_features.parquet"
    max_feat = 300
    tf_idf_file_path = data_path/ f"tfidf_features_{max_feat}.parquet"
    
    if tf_idf_file_path.exists():
        print(f"File exists: {tf_idf_file_path}")
    else:
        print(f"File not found: {tf_idf_file_path}")
        save_tfidf_features(clinical_texts, data_path, max_features=max_feat)
    
    if bert_file_path.exists():
        print(f"File exists: {bert_file_path}")
    else:
        print(f"File not found: {bert_file_path}")
        save_bert_features(clinical_texts, data_path)

    #fix_data(text_data=clinical_texts)

