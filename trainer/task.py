# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
import torch
import argparse
import hypertune
import numpy as np
from transformers import (AutoTokenizer,Trainer,TrainingArguments,default_data_collator, AutoModelForSequenceClassification)
from trainer import utils

def init_args():
    # Experiment arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--batch-size', type=int, default=16)
    args_parser.add_argument('--num-epochs',type=int,default=1)
    args_parser.add_argument( '--learning-rate', default=2e-5, type=float)
    args_parser.add_argument('--weight-decay', default=0.01, type=float)
    args_parser.add_argument('--job-dir', default=os.getenv('AIP_MODEL_DIR'), help='GCS location to export models')
    args_parser.add_argument( '--model-name', default="finetuned-bert-sentiment-analysis")
    return args_parser.parse_args()


def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


def train(args, model, train_dataset, test_dataset):
    # initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(utils.NAME_BERT,use_fast=True,)

    # set training arguments
    training_args = TrainingArguments(
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        output_dir=os.path.join("/tmp", args.model_name)
    )

    # initialize our Trainer
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer


def run(args):
    # load dataset
    train_dataset, test_dataset = utils.load_data(args)
    label_list = train_dataset.unique("label")
    num_labels = len(label_list)

    # init model
    model_bert = AutoModelForSequenceClassification.from_pretrained(utils.NAME_BERT,num_labels=num_labels)
    trainer = train(args, model_bert, train_dataset, test_dataset)

    # Export the trained model
    trainer.save_model(os.path.join("/tmp", args.model_name))

    # Save the model to GCS
    if args.job_dir:
        utils.save_model(args)
    else:
        print(f"Saved model files at {os.path.join('/tmp', args.model_name)}")

def predict(input_text, path_model_local):
    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(utils.NAME_BERT, use_fast=True)

    tokenizer_args = (input_text,)
    predict_input = tokenizer( *tokenizer_args, padding="max_length",
        max_length=utils.MAX_SEQ_LENGTH, truncation=True, return_tensors="pt",)

    # load trained model
    loaded_model = AutoModelForSequenceClassification.from_pretrained(path_model_local)
    output = loaded_model(predict_input["input_ids"])
    label_id = torch.argmax(*output.to_tuple(), dim=1)

    print(f"Review text: {input_text}")
    print(f"Sentiment : {utils.label_text[label_id.item()]}\n")

def main():
    args = init_args()
    run(args)


if __name__ == '__main__':
    main()
