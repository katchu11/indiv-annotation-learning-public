import argparse
import pickle
import torch
from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments, RobertaForSequenceClassification
import wandb
from openai import OpenAI

from models import RobertaMultitaskModel, OpenAITextModel, OpenAITextAnnotatorModel

client = OpenAI() # add key

class CustomTrainer(Trainer):
    def evaluate(self,
                 eval_dataset=None,
                 ignore_keys=None,
                 metric_key_prefix: str = "eval",
                 return_preds=True):
        """
        Evaluate the annotator rating model.
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.eval()

        dataloader = self.get_eval_dataloader(eval_dataset)
        num_batches = len(dataloader)

        all_preds = []
        all_labels = []
        for batch_index, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                if isinstance(self.model, (OpenAITextModel, OpenAITextAnnotatorModel)):
                    texts = batch["input_texts"]
                    if isinstance(self.model, OpenAITextAnnotatorModel):
                        annotators = batch["annotators"]
                        outputs = self.model(texts, annotators)
                    else:
                        outputs = self.model(texts)
                else:
                    outputs = self.model(**batch)

            logits = outputs.logits if not isinstance(outputs, torch.Tensor) else outputs
            predictions = torch.sigmoid(logits).detach().cpu().numpy()
            predictions = [1.0 if x > 0.5 else 0.0 for x in predictions]

            all_preds.extend(predictions)
            all_labels.extend(batch["labels"].detach().cpu().numpy())

        accuracy = sum(1 for x, y in zip(all_preds, all_labels) if x == y) / len(all_preds)
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy

def load_model_and_tokenizer(model_path, lm_version):
    if lm_version == 'roberta':
        model_config = AutoConfig.from_pretrained("roberta-base")
        tokenizer = AutoTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
        model = RobertaMultitaskModel.from_pretrained(model_path, config=model_config)
    elif lm_version == 'openai_text':
        model = OpenAITextModel()
        tokenizer = None  # OpenAI embeddings do not require a tokenizer
    elif lm_version == 'openai_text_annotator':
        model = OpenAITextAnnotatorModel()
        tokenizer = None  # OpenAI embeddings do not require a tokenizer
    else:
        raise ValueError("Unsupported lm_version")

    return model, tokenizer

def evaluate_model(model, tokenizer, test_data_path, batch_size):
    test_dataset = load_from_disk(test_data_path)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=batch_size,
        report_to="none"
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset
    )

    return trainer.evaluate(test_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to the tokenized test dataset')
    parser.add_argument('--lm_version', choices=['roberta', 'openai_text', 'openai_text_annotator'], default='roberta', help='Language model version')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.lm_version)
    evaluate_model(model, tokenizer, args.test_data_path, args.batch_size)
