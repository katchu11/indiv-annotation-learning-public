import json
import os
import re
from transformers import AutoTokenizer, pipeline
import torch
from collections import Counter
import datasets
import pandas as pd


# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)
model_paths = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    # "meta-llama/Llama-2-7b-chat-hf",
    # "meta-llama/Llama-2-13b-chat-hf"
]

# Function to initialize model pipeline
def init_model_pipeline(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path,padding_side='left')
    model_pipeline = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model_pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id
    return model_pipeline

# Function to format the prompt
def format_prompt(prompt):
    historical_annotations = [f"[{text}] is rated as [{rating}]" for text, rating in prompt["history"]]
    historical_text = "The annotator has also annotated these texts from a scale of 0 to 4, where 0 is the least toxic and 4 is the most toxic: " + ', '.join(historical_annotations)
    combined_text = f"{historical_text} [SEP] {prompt['post']}"
    
    return {
        "messages": [
            {"role": "system", "content": "You are a model that predicts the toxicity rating of text from 0 to 4, where 0 is the least toxic and 4 is the most toxic."},
            {"role": "user", "content": combined_text}
        ]
    }
# Function to generate a response
def generate_response(prompt, model_pipeline):
    sequences = model_pipeline(
        prompt['messages'],
        max_new_tokens=64,
        do_sample=True,
        top_k=10,
        temperature=0.2,
        num_return_sequences=1,
        eos_token_id=model_pipeline.tokenizer.eos_token_id,
        pad_token_id=model_pipeline.tokenizer.pad_token_id,
        batch_size=8
    )
    return {'response': [sequences[i][0]['generated_text'] for i in range(len(sequences))]}

# Main function to process files and generate responses for each model
def process_files_for_models():
    model_path = model_paths[0]
    print(f"Processing for model: {model_path}")
    model_pipeline = init_model_pipeline(model_path)
    
    data = pd.read_csv("toxjson_df.csv").sample(1000)      
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=data))
    dataset = dataset.map(format_prompt)
    results = dataset.map(lambda x: generate_response(x, model_pipeline), batched=True, batch_size=8)
    answers = [extract_answer(r) for r in results]
    with open(f'./results/{model_path.split("/")[-1]}.json', 'w') as f:
        json.dump(answers, f)
    total_dev = 0
    cntr = 0
    for index,answer in enumerate(answers):
        if answer != -1:
            total_dev += abs(data.iloc[index]["labels"] - answer)
            cntr +=1
    print("MAE:",total_dev/cntr)
    print(total_dev,cntr)
    print("pct predicted", cntr/len(data))
def extract_answer(data):
    
    # Splitting the data to only consider the part after [/INST]
    post_inst_data = data.split("[/INST]", 1)[1] if "[/INST]" in data else ""
    try:
        rating = int(post_inst_data)
    except:
        rating= -1
    # Constructing the output JSON
    extracted_info = {
        "Rating":  rating   }
    return extracted_info

if __name__ == '__main__':
    process_files_for_models()
