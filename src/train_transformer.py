import warnings
import random
from transformers import logging
from datasets import Dataset, DatasetDict, Features, ClassLabel, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from sklearn.model_selection import train_test_split

# Suppress expected warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def simple_augment(text):
    """Simple text augmentation without external dependencies"""
    augmentation_techniques = {
        0: lambda t: ''.join(random.choice([c.upper(), c.lower()]) for c in t),
        1: lambda t: t[:-1] if t[-1] in '.!?' else t + random.choice(['.', '!', '?']),
        2: lambda t: (words := t.split()) and len(words) > 1 and 
                    (words.insert(idx := random.randint(0, len(words)-1), words[idx]) and ' '.join(words))
    }
    return augmentation_techniques.get(random.randint(0, 2), lambda t: t)(text)

def load_and_prepare_dataset():
    try:
        df = pd.read_csv('data/training_data.csv')
        if {'text', 'label'} - set(df.columns):
            raise ValueError("CSV must contain 'text' and 'label' columns")
        
        if len(df[df['label'] == 1]) < 2 or len(df[df['label'] == 0]) < 2:
            raise ValueError("Need at least 2 examples of each class")
        
        if len(df) < 100:
            print(f"\nWarning: Very small dataset ({len(df)} examples)\nRecommended minimum is 1000\n")
        
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
        
        features = Features({'text': Value('string'), 'label': ClassLabel(names=['clean', 'toxic'])})
        
        return DatasetDict({
            'train': Dataset.from_dict({'text': train_df['text'].tolist(), 'label': train_df['label'].tolist()}, features=features),
            'test': Dataset.from_dict({'text': test_df['text'].tolist(), 'label': test_df['label'].tolist()}, features=features)
        })
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def train_and_save_model():
    dataset = load_and_prepare_dataset()
    
    if len(dataset['train']) < 100:
        print("Applying simple data augmentation...")
        dataset['train'] = dataset['train'].add_column(
            'augmented_text', 
            [simple_augment(text) for text in dataset['train']['text']]
        )
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label={0: "clean", 1: "toxic"}, label2id={"clean": 0, "toxic": 1}
    )    
    def tokenize_function(examples):
        text_column = 'augmented_text' if 'augmented_text' in examples else 'text'
        return tokenizer(
            list(map(str, examples[text_column])),  # Ensure all elements are strings
            padding="max_length",
            truncation=True,
            max_length=128
        )

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True
    )    
    training_args = TrainingArguments(
        output_dir='models/transformer_model',
        per_device_train_batch_size=4 if len(dataset['train']) < 100 else 16,
        per_device_eval_batch_size=4,
        num_train_epochs=10 if len(dataset['train']) < 100 else 4,
        eval_strategy="epoch" if len(dataset['train']) < 100 else "steps",
        eval_steps=50,
        save_strategy="epoch" if len(dataset['train']) < 100 else "steps",
        save_steps=50,
        learning_rate=5e-5 if len(dataset['train']) < 100 else 2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=42,
        gradient_accumulation_steps=4 if len(dataset['train']) < 100 else 2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("Starting training...")
    trainer.train()
    
    print("\nFinal evaluation results:")
    for key, value in trainer.evaluate().items():
        print(f"{key}: {value:.4f}")

    model.save_pretrained("models/transformer_model")
    tokenizer.save_pretrained("models/transformer_model")
    print("\nModel saved successfully!")

if __name__ == "__main__":
    train_and_save_model()