import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, DatasetDict, Audio
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
# 1. Load Dataset (Mozilla Common Voice 11.0 - Tamil)
# ==========================================
print("Loading dataset...")
common_voice = DatasetDict()

# Load train and test splits. Note: requires Hugging Face authentication to download.
# Make sure you have run `huggingface-cli login` if needed.
common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "ta", split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "ta", split="test", use_auth_token=True)

# Remove unnecessary columns
common_voice = common_voice.remove_columns([
    "accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"
])

# ==========================================
# 2. Processor, Feature Extractor, Tokenizer
# ==========================================
model_name = "openai/whisper-small"
print(f"Loading processor for {model_name}...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Tamil", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_name, language="Tamil", task="transcribe")

# ==========================================
# 3. Resample Audio Data
# ==========================================
print("Resampling audio to 16kHz...")
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

# ==========================================
# 4. Prepare Dataset Function
# ==========================================
def prepare_dataset(batch):
    audio = batch["audio"]
    # Compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # Encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

print("Preparing dataset with features and labels...")
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)

# ==========================================
# 5. Data Collator
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
        
        # If bos token is appended in previous tokenization step, cut bos token here.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# ==========================================
# 6. Evaluation Metric (WER)
# ==========================================
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ==========================================
# 7. Model
# ==========================================
print("Loading model...")
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Disable cache during training
model.config.use_cache = False
# Generate in Tamil
model.generate = partial(model.generate, language="tamil", task="transcribe")

# ==========================================
# 8. Training Configuration
# ==========================================
training_args = Seq2SeqTrainingArguments(
    output_dir="./model",  
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available(),
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# ==========================================
# 9. Train and Save
# ==========================================
print("Starting training...")
trainer.train()

print("Training complete. Saving processor and model...")
processor.save_pretrained(training_args.output_dir)
model.save_pretrained(training_args.output_dir)
print("Saved to", training_args.output_dir)
