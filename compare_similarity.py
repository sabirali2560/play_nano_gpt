import argparse
import os
import nltk
from nltk.tokenize import word_tokenize
import random

nltk.download('punkt')

input_file = 'input.txt'
input_file_path_sk = os.path.join('data', 'shakespeare_char', input_file)
input_file_path_bd = os.path.join('data', 'breaking_bad_char', input_file)
original_text_sk = ""
original_text_bd = ""

#read shakespeare input file
with open(input_file_path_sk, 'r') as f:
    original_text_sk = f.read()
print(f"length of input dataset in characters for shakespeare: {len(original_text_sk):,}")

#read breaking bad input file
with open(input_file_path_bd, 'r') as f:
    original_text_bd = f.read()
print(f"length of input dataset in characters for breaking bad: {len(original_text_bd):,}")

output_file = 'output.txt'
output_file_path = os.path.join('data', 'breaking_bad_char', output_file)

generated_text = ""

#read output file
with open(output_file_path, 'r') as f:
    generated_text = f.read()
print(f"length of output dataset in characters: {len(generated_text):,}")


# Tokenize the reference and generated texts into lists of words
original_tokenized_sk = word_tokenize(original_text_sk)
original_tokenized_bd = word_tokenize(original_text_bd)

generated_tokenized = word_tokenize(generated_text)
original_tokenized_sk = random.sample(original_tokenized_sk, len(generated_tokenized))
original_tokenized_bd = random.sample(original_tokenized_bd, len(generated_tokenized))

weights = (1.0, 0.0, 0.0, 0.0)

# Calculate the BLEU score
bleu_score_sk = nltk.translate.bleu_score.sentence_bleu([original_tokenized_sk], generated_tokenized, weights)

bleu_score_bd = nltk.translate.bleu_score.sentence_bleu([original_tokenized_bd], generated_tokenized, weights)

print("BLEU Score for Shakespeare:", bleu_score_sk)

print("BLEU Score for Breaking Bad:", bleu_score_bd)
