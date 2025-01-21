from openai import OpenAI
import torch
from torch import nn
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments, TrainerCallback
from sklearn.metrics import mean_absolute_error

# Set your OpenAI API key
client = OpenAI(api_key="***")


# Load dataset from disk
dataset = load_from_disk('./DATA_ROOT/multitasktext-sepTrue/train')

# Function to generate embeddings using OpenAI
def embed_text(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Extract text after the last '[SEP]' token
def extract_text_after_sep(text):
    return text.split('[SEP]')[-1]

# Generate historical annotation summary for each annotator
def generate_historical_summary(dataset):
    ann_dict = {}
    for example in dataset:
        ann_id = example['ann_ids']
        text = extract_text_after_sep(example['post'])
        rating = example['labels']
        if ann_id not in ann_dict:
            ann_dict[ann_id] = []
        ann_dict[ann_id].append((text, rating))
    
    historical_summaries = {}
    for ann_id, annotations in ann_dict.items():
        summary = "The annotator has also annotated these texts from a scale of 0 to 4, where 0 is the least toxic and 4 is the most toxic: "
        summaries = [f"[{text}] is rated as [{rating}]" for text, rating in annotations]
        historical_summaries[ann_id] = summary + ', '.join(summaries)
    
    return historical_summaries

# Generate historical summaries
historical_summaries = generate_historical_summary(dataset)

# Add embeddings to the dataset with historical information
def process_example(example):
    ann_id = example['ann_ids']
    historical_text = historical_summaries[ann_id]
    current_text = example['post']
    combined_text = f"{historical_text} [SEP] {current_text}"
    embedding = embed_text(combined_text)
    return {
        'embedding': embedding,
        'label': example['label']
    }

dataset = dataset.map(process_example)

# Convert embeddings and labels to the correct format
def format_dataset(examples):
    return {
        'embedding': examples['embedding'],
        'label': examples['label']
    }

dataset = dataset.map(format_dataset, batched=True)

# Model definition
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)  # Assuming regression task

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleModel()

# Convert the dataset to PyTorch tensors
def convert_to_torch(dataset):
    embeddings = torch.tensor([item['embedding'] for item in dataset], dtype=torch.float32)
    labels = torch.tensor([item['label'] for item in dataset], dtype=torch.float32)
    return {'input_ids': embeddings, 'labels': labels}

train_dataset = convert_to_torch(dataset)

# Check if the validation dataset exists
eval_dataset = convert_to_torch(load_from_disk("./DATA_ROOT/multitasktext-sepTrue/dev"))

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=6,
    per_device_train_batch_size=32,
    logging_dir='./logs',
    evaluation_strategy="epoch" if eval_dataset else "no"
)

# Define the MAE compute function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()  # Remove the extra dimension
    mae = mean_absolute_error(labels, predictions)
    return {"mae": mae}

# Custom callback to compute training MAE
class ComputeTrainMAECallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.train_losses.append(logs['loss'])

    def on_epoch_end(self, args, state, control, **kwargs):
        # Compute MAE on the training dataset
        model.eval()
        train_preds = []
        train_labels = []
        for batch in train_dataset:
            inputs = batch['input_ids']
            labels = batch['labels']
            with torch.no_grad():
                outputs = model(inputs)
                train_preds.extend(outputs.squeeze().tolist())
                train_labels.extend(labels.tolist())

        mae = mean_absolute_error(train_labels, train_preds)
        print(f"Training MAE after epoch {state.epoch}: {mae}")

# Initialize the Trainer with the custom callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics if eval_dataset else None,
    callbacks=[ComputeTrainMAECallback()]  # Add the custom callback here
)
print('starting training')
# Train the model
trainer.train()

# Evaluate the model if eval_dataset is provided
if eval_dataset:
    eval_results = trainer.evaluate()
    print(f"MAE: {eval_results['eval_mae']}")
else:
    print("No evaluation dataset provided.")
