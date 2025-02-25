"""
**Automated Calling of LLMs**
This script conducts automated calls of the LLMs with our prompt for 439 games.

*   Calls Llama3-70b-instruct-v1:0

Assumes downloaded: prompt.txt, all_games.txt
Dependencies: transformers, torch, pandas (see requirements.txt file)
"""

# import openai
import random
# import boto3
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import csv
# import anthropic
# import google.generativeai as genai
import pandas as pd
from tqdm import tqdm

# Add your api keys
# openai.api_key = ""
# gemini_api_key = ""
# claude_api_key = ""
# llama_api_key = ""

# client = boto3.client('bedrock-runtime',region_name="us-east-1")
# model_id = 'meta.llama3-70b-instruct-v1:0'
# This path goes to a better but larger model => "/datasets/ai/llama3/meta-llama/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/33101ce6ccc08fa6249c10a543ebfcac65173393"
path_to_model = "/datasets/ai/llama3/meta-llama/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6/" 
tokenizer = AutoTokenizer.from_pretrained(path_to_model)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(path_to_model, device_map="auto")
device = "cuda" if torch.cuda.is_available() else "cpu"

def open_games(games_file):
    """Open games file, read each game onto a new line, and shuffle game.
    """
    games_df = pd.read_csv(games_file)
    # Read answers into 4 lists of 4 words
    # games_df["Words"] = games_df["Words"].apply(split_and_reshape)
    # games_list = games_df["Words"].tolist()
    games_df["Input"] = games_df["Input"].apply(split_and_reshape)
    games_list = games_df["Input"].tolist()
    return games_list

def split_and_reshape(game):
    """ Read rows of csv in into list, shuffle, and then convert back to string
    """
    game = game.replace("'", "")
    game = game.replace('[', "")
    game = game.replace(']', "")
    # Convert words to list of words
    game = list(game.split(", "))
    # Shuffle words
    random.shuffle(game)
    # Convert to string
    game = ', '.join(game)
    return game

def open_prompt(prompt_file):
    """Open prompt text file.
    """
    prompt_file = open(prompt_file, 'r')
    return prompt_file.read()

# def run_chatgpt(prompt, api_key):
#     response = openai.chat.completions.create(
#         model="gpt-4o",
#         messages=[{"role": "system", "content": prompt}, ],
#         max_tokens=750,
#     )
#     return response.choices[0].message.content

# def run_gemini(prompt, api_key):
#     genai.configure(api_key=api_key)
#     model = genai.GenerativeModel('gemini-1.5-pro')
#     response = model.generate_content(prompt)
#     return response.text

# def run_claude(prompt, api_key):
#     client = anthropic.Anthropic(api_key=api_key,)
#     message = client.messages.create(
#         model="claude-3-opus-20240229",
#         max_tokens=750,
#         messages=[{"role": "user", "content": prompt}],
#     )
#     return message.content

def messages2llama3str(messages):
    formatted_str = "<|begin_of_text|>"
    for message in messages:
        role = message["role"]
        content = message["content"]
        formatted_str += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    formatted_str += "<|start_header_id|>assistant<|end_header_id|>"
    return formatted_str

# def run_llama3(game_prompt):
#     messages = [{"role": "user", "content": game_prompt}]
#     prompt = messages2llama3str(messages)
#     request = {"prompt": prompt,"max_gen_len": 750,"temperature": 0.6,"top_p": 0.9}
#     response = client.invoke_model(contentType='application/json', body=json.dumps(request), modelId=model_id)
#     inference_result = response['body'].read().decode('utf-8')
#     inference_result = json.loads(inference_result)['generation']
#     return inference_result
def run_llama3(game_prompt):
    """Run LLaMA 3 using Hugging Face Transformers."""
    messages = [{"role": "user", "content": game_prompt}]
    prompt = messages2llama3str(messages)
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Generate response
    with torch.no_grad():
        # output = model.generate(
        #     **inputs,
        #     max_length=750,
        #     temperature=0.6,
        #     top_p=0.9
        # )
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    # Decode generated text
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    clean_text = response_text.split("Example 4:\n")[1]
    return clean_text


# def run_games(games, filename, play, api_key):
def run_games(games, filename, play, maxGames=float('inf')):
    """Run model (based on play function) with prompt.
    New game will be inserted into prompt string.
    """
    with open(filename, mode='w', newline='') as file:
        # Create a csv.writer object
        writer = csv.writer(file)
        # Exclude demonstration games that appear in prompt (first 3 rows) and run 200 games
        for n in tqdm(range(min(len(games), maxGames))):
        # for n in range(200):
            game_prompt = prompt.replace("InsertGame", games[n])
            # response = play(game_prompt, api_key)
            response = play(game_prompt)
            response.replace("\n", " ")
            writer.writerow([response])


if __name__ == '__main__':
    # games_list = open_games('connectionsRes.csv')
    games_list = open_games('scoring/scripts/beginner.csv')
    # prompt = open_prompt("prompt.txt")
    prompt = open_prompt("automated_call/prompt_llm.txt")
    # run_games(games_list, "chatgpt_responses.csv", run_chatgpt, openai.api_key)
    # run_games(games_list, "claude3_responses.csv", run_claude, claude_api_key)
    # you need AWS api key to run this
    run_games(games_list, "llama3_responses.csv", run_llama3, maxGames=2)
