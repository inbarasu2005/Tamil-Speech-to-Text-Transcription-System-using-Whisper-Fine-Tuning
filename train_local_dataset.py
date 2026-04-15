import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pandas as pd
from datasets import Dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
from functools import partial

# ==========================================
# 1. Load Custom Local Dataset
# ==========================================
# Make sure you have a metadata.csv file inside the dataset folder
# with two columns: "file_name" and "sentence"
# And a folder containing your .wav files.
path_to_csv = "dataset/metadata.csv"

print("Loading local custom dataset...")
df = pd.read_csv(path_to_csv)

# Rename columns if necessary so that datasets can use them
# Ensure your CSV has 'file_name' (path to the audio file) and 'sentence' (the Tamil text)
dataset = Dataset.from_pandas(df)

# Cast the file_name column to an Audio column so datasets library loads the actual audio data
dataset = dataset.cast_column("file_name", Audio(sampling_rate=16000))

# Split into train and test sets
split_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

# ==========================================
# 2. Processor, Feature Extractor, Tokenizer
# ==========================================
model_name = "openai/whisper-small"
print(f"Loading processor for {model_name}...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Tamil", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_name, language="Tamil", task="transcribe")

# ==========================================
# 3. Prepare Dataset Function
# ==========================================
def prepare_dataset(batch):
    # 'file_name' was casted to Audio, so it contains the audio array and sampling rate
    audio = batch["file_name"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

print("Preparing dataset with features and labels...")
train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, num_proc=1)
test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names, num_proc=1)

# ==========================================
# 4. Data Collator
# ==========================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# ==========================================
# 5. Evaluation Metric (WER)
# ==========================================
metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ==========================================
# 6. Model
# ==========================================
print("Loading model...")
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.config.use_cache = False
model.generate = partial(model.generate, language="tamil", task="transcribe")

# ==========================================
# 7. Training Configuration
# ==========================================
training_args = Seq2SeqTrainingArguments(
    output_dir="./model",  
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=1000,
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available(),
    evaluation_strategy="steps",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=200,
    eval_steps=200,
    logging_steps=10,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# ==========================================
# 8. Train and Save
# ==========================================
print("Starting training on local dataset...")
trainer.train()

print("Training complete. Saving processor and model...")
processor.save_pretrained(training_args.output_dir)
model.save_pretrained(training_args.output_dir)
print("Saved to", training_args.output_dir)
