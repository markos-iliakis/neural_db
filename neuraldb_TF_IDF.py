import json
from datasets import Dataset, DatasetDict
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import Trainer, TrainingArguments
from sklearn.metrics.pairwise import cosine_similarity
import spacy

nlp = spacy.load("en_core_web_lg")


def load_data():
    train_dbs, test_dbs, dev_dbs = [], [], []

    # Load data
    with open('./Data/v2.4_25/train.jsonl', 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        train_dbs.append(result)

    with open('./Data/v2.4_25/dev.jsonl', 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        dev_dbs.append(result)

    with open('./Data/v2.4_25/test.jsonl', 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        test_dbs.append(result)

    return train_dbs, test_dbs, dev_dbs


def tokenize_function(examples):
    print(examples)
    return tokenizer(examples["query"], examples["facts"], padding="max_length", truncation=True)


def tf_idf_sk(query, facts):
    query = ' '.join([token.lemma_.lower() for token in nlp(query) if not token.is_stop and not token.is_punct])
    n_facts = list()
    for fact in facts:
        n_facts.append(' '.join([token.lemma_.lower() for token in nlp(fact) if not token.is_stop and not token.is_punct]))

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query] + n_facts)

    scores = list()
    for i in range(1, len(n_facts)+1):
        scores.append([cosine_similarity(vectors[0], vectors[i]), n_facts[i-1]])

    return scores


if __name__ == '__main__':

    MODEL_NAME = "t5-base"

    # Create the model and the tokenizer
    model = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load data
    train_dbs, test_dbs, dev_dbs = load_data()

    train_dataset = []
    for db in train_dbs:
        for query in db['queries']:
            # Get the query and its facts and rank them based on tfidf
            query_text = query['query']
            query_facts = tf_idf_sk(query_text, db['facts'])
            # query_facts = [[tf_idf_sk(query_text, fact), fact] for fact in db['facts']]
            query_facts = sorted(query_facts, key=lambda x: x[0], reverse=True)

            # Keep only the n top facts
            top_facts = query_facts[:3]

            text = ''
            for fact in top_facts:
                text += fact + ' '

            text = text[:-1]
            train_dataset.append([query_text, text])

    df = pd.DataFrame(train_dataset, columns=['query', 'facts'])
    td = Dataset.from_pandas(df)
    dataset_dict = {"train": td,
                    "test": td,
                    "unsupervised": td}

    d = DatasetDict(dataset_dict)

    # Tokenize datasets
    tokenized_datasets = d.map(tokenize_function, batched=True)

    args = TrainingArguments(
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        num_train_epochs=3,
        weight_decay=0.01,
        output_dir="./"
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets.get("train"),
        eval_dataset=tokenized_datasets.get("test")
    )

    trainer.train()
