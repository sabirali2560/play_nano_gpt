import argparse
import os
import nltk
from nltk.tokenize import word_tokenize
import random

nltk.download('punkt')

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="A script that evaluates bleu score between input and output files.")

# Add command-line arguments with specific flag names
parser.add_argument('--input_file', '-i', type=str, help="input file name.")
parser.add_argument('--output_file', '-o', type=str, help="Path to output file name.")

# Parse the command-line arguments
args = parser.parse_args()

input_file_path = os.path.join('data', 'shakespeare_char', args.input_file)
original_text = ""

#read input file
with open(input_file_path, 'r') as f:
    original_text = f.read()
print(f"length of dataset in characters: {len(original_text):,}")

output_file_path = os.path.join('data', 'shakespeare_char',args.output_file)

generated_text = ""

#read output file
with open(output_file_path, 'r') as f:
    generated_text = f.read()
print(f"length of dataset in characters: {len(generated_text):,}")


# Tokenize the reference and generated texts into lists of words
original_tokenized = word_tokenize(original_text)

generated_tokenized = word_tokenize(generated_text)
original_tokenized = random.sample(original_tokenized, len(generated_tokenized))

print(len(original_tokenized))

weights = (1.0, 0.0, 0.0, 0.0)

# Calculate the BLEU score
bleu_score = nltk.translate.bleu_score.sentence_bleu([original_tokenized], generated_tokenized, weights)

print("BLEU Score:", bleu_score)
