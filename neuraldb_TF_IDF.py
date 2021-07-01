import json
from datasets import Dataset, DatasetDict
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import Trainer, TrainingArguments
import sklearn


def text_preprocess(text):
    stop = stopwords.get_stopwords('english')
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    lemmatizer = WordNetLemmatizer()

    # Make lowercase
    text = np.char.lower(text).tolist()

    # Remove stop words
    new_text = ""
    for word in text:
        if word not in stop:
            new_text = new_text + " " + word
    text = new_text

    # Remove symbols
    for i in symbols:
        text = np.char.replace(text, i, ' ')

    # Replace apostrophe
    text = np.char.replace(text, "'", "").tolist()

    # Remove single characters
    new_text = ""
    for w in text:
        if len(w) > 1:
            new_text = new_text + " " + w
    text = new_text

    # Lemmatize words
    new_text = ""
    for w in text:
        new_text = new_text + lemmatizer.lemmatize(w)
    text = new_text

    return text


def compute_tf(word_dict, bow):
    tf_dict = {}
    bow_count = len(bow)
    for word, count in word_dict.items():
        tf_dict[word] = count / float(bow_count)
    return tf_dict


def compute_idf(documents):
    import math
    N = len(documents)

    idf_dict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idf_dict[word] += 1

    for word, val in idf_dict.items():
        idf_dict[word] = math.log(N / float(val))
    return idf_dict


def tf_idf(query, fact):
    # Preprocess text
    bow_query = query.split(' ')
    bow_fact = fact.split(' ')

    bow_query = text_preprocess(bow_query)
    bow_fact = text_preprocess(bow_fact)

    unique_words = set(bow_query).union(set(bow_fact))

    num_words_query = dict.fromkeys(unique_words, 0)
    for w in bow_query:
        num_words_query[w] += 1

    num_words_fact = dict.fromkeys(unique_words, 0)
    for w in bow_fact:
        num_words_fact[w] += 1

    tf_query = compute_tf(num_words_query, bow_query)
    tf_fact = compute_tf(num_words_fact, bow_fact)

    idfs = compute_idf([num_words_query, num_words_fact])

    return 1


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


def tf_idf_sk(query, fact):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query, fact])
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)

    return df.values.sum()/df.shape[1]


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
