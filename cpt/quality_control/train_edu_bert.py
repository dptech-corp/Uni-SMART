from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset, ClassLabel, Dataset
import numpy as np
import evaluate
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix
import json
from multiprocessing import cpu_count
import pandas as pd

def write_jsonl(data, filename):
    with open(filename, 'w') as f:
        lines = [json.dumps(d)+'\n' for d in data]
        f.writelines(lines)

def read_jsonl(filename):
    with open(filename, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    return data

def compute_metrics(eval_pred):
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    mae_metric = evaluate.load("mae")
    mse_metric = evaluate.load("mse")
    r2_metric = evaluate.load("r_squared")

    logits, labels = eval_pred
    preds = np.round(logits.squeeze()).clip(0, 5).astype(int)
    float_preds = logits.squeeze()
    labels = np.round(labels.squeeze()).astype(int)
    precision = precision_metric.compute(predictions=preds, references=labels, average="macro")["precision"]
    recall = recall_metric.compute(predictions=preds, references=labels, average="macro")["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    mae = mae_metric.compute(predictions=float_preds, references=labels)["mae"]
    mse = mse_metric.compute(predictions=float_preds, references=labels)["mse"]
    r2 = r2_metric.compute(predictions=float_preds, references=labels)

    report = classification_report(labels, preds)
    cm = confusion_matrix(labels, preds)
    print("Validation Report:\n" + report)
    print("Confusion Matrix:\n" + str(cm))
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "accuracy": accuracy,
        "mae": mae,
        "mse": mse,
        "r2": r2,
    }
    
    print(json.dumps(metrics, indent=4))

    return metrics


def main(args):
    data_dict = read_jsonl(args.dataset_path)
    # data_dict = data_dict[:1000]
    dataset = Dataset.from_list(data_dict)
    dataset = dataset.map(
        lambda x: {args.target_column: np.clip(int(x[args.target_column]), 0, 5)}, num_proc=cpu_count()
    )

    dataset = dataset.cast_column(
        args.target_column, ClassLabel(names=[str(i) for i in range(6)])
    )
    dataset = dataset.train_test_split(
        train_size=len(dataset)-10 if args.no_eval else 0.9, seed=42, stratify_by_column=args.target_column
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    def preprocess(examples):
        batch = tokenizer(examples["text"], truncation=True)
        batch["labels"] = np.float32(examples[args.target_column])
        return batch

    dataset = dataset.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model_name, num_labels=1, classifier_dropout=0.0, hidden_dropout_prob=0.0)

    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for param in model.bert.encoder.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=10000,
        logging_steps=5,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        seed=0,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=128,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    if not args.no_train:
        trainer.train()

        model_save_path = os.path.join(args.checkpoint_dir, "final")
        trainer.save_model(model_save_path)
        print(f"Model saved to {model_save_path}")

        log_save_path = os.path.join(args.checkpoint_dir, "log_history.csv")
        log_history_df = pd.DataFrame(trainer.state.log_history)
        log_history_df.to_csv(log_save_path, index=False)
        print(f"Log history saved to {log_save_path}")

    if not args.no_eval:
        evaluation_results = trainer.evaluate()
        evaluation_results_path = os.path.join(args.checkpoint_dir, "evaluation_results.json")
        with open(evaluation_results_path, "w") as f:
            json.dump(evaluation_results, f, indent=4)
        print(f"Evaluation results saved to {evaluation_results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument("--base_model_name", type=str, default="HuggingFaceFW/fineweb-edu-classifier")
    parser.add_argument("--no_train", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--dataset_path", type=str, default="none")
    parser.add_argument("--target_column", type=str, default="score")
    parser.add_argument("--checkpoint_dir", type=str, default="none")
    # model parameters
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--per_device_train_batch_size", type=int, default=256)
    args = parser.parse_args()
    print('-'*200)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print('-'*200)
    main(args)