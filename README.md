# Playing with Nano GPT

This repo is my attempt to experiment with a few aspects of GPT and getting hands on experience of all my theoretical learnings. I have used Karpthy's version of a nano GPT for this experimentation. You can find more info about the same here : https://github.com/karpathy/nanoGPT

I focused on the following tasks :

## Setup and Reproduction

Setting up the basic flow for our experiment and generating some initial results out of it

Prepare 

```
python data/shakespeare_char/prepare.py
```
Train

```
python train.py config/train_shakespeare_char.py --device=mps --compile=False
```

Sample

```
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

Example generated by the model

```
BRUTUS:
For the devil'd the beast torm:
When could I should be saw the pride
That should be thou not be subject.'

MERCUTIO:
For what, comes the way
Methink of my company?

MERCUTIO:
Tranio, Romeo, go, tyrant, and since to speak.

SIRREY:
Then did your hearts first,
For I make more call them again.

BRUTUS:
Come, sir, my lord.

SICINIUS:
Sir, sir.

CORIOLANUS:
Well, let us murderer?

First Servingman:
Take me to have better.

First Citizen:
I can perfort you are thou wert not the good?

CORIOLAN
```

## Hyperparameter Experimentation

Modifying hyperparameters such as number of heads, layers in order to achieve a setting which produces the lowest validation loss.

Use the following command to train nano GPT on Shakespeare data using your Mac's on-chip GPU. Using lower settings for hyperparameter so that it doesn't take more than 10 mins to run.
Feel free to play around with these hyperparameters.

```
python train.py config/train_shakespeare_char.py --device=mps --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=3000 --lr_decay_iters=3000 --dropout=0.0
```
We get the following losses with different number of heads and layers :

1. layers = 4, heads = 4 : train loss 1.6169, val loss 1.7946
2. layers = 8, heads = 8 : train loss 1.6393, val loss 1.7759
3. layers = 16, heads = 16 : train loss 1.5978, val loss 1.7658
4. layers = 32, heads = 32 : train loss 1.5765, val loss 1.7662

## Evaluation Metrics

### Specific Metric : Meant to capture how close the generated data distribution is to the training distribution.

I have used Bleu score for my specific metric. The BLEU (Bilingual Evaluation Understudy) score is computed by comparing n-grams (contiguous sequences of n words or characters) between the generated text and reference texts. BLEU typically considers n-grams of different lengths, from unigrams (single words) to higher-order n-grams like bigrams, trigrams, and so on. Using this kind of evaluation BLEU measures the similarity between the generated text and the reference text (training text). This seems ideal for our current use case of comparing how close the generated data distribution is to the training distribution. 

I wrote a script evaluation_bleu.py which uses the nltk library in order to compute the bleu score.

```
python evaluation_bleu.py --data_dir=shakespeare_char
```

Result : ``` BLEU Score: 0.5455284552845528 ```

### General Metric : Meant to capture how our model performs in general for text generation regardless of data it has been trained on.

I have used a simple spell check function to test if our model produces words which actually exist in the English language. This makes sense as we are training a character level GPT and if our model is able to associate characters into meaningful words we are in a win-win situation. 

I wrote a script evaluation_spell_check.py which uses the nltk words corpus for spell checking the generated words by our model.

```
python evaluation_spell_check.py --data_dir=shakespeare_char
```
Result : ``` Percentage of words not correctly spelled:  8.292682926829269 % ```

## My favourite Dataset

Here I experiment training nano GPT with my favourite dataset which is the screenplay scripts from the popular TV series Breaking Bad

I have downloaded this dataset from : https://bulletproofscreenwriting.tv/breaking-bad-tv-script-download/

In particular I use scripts from the following episodes:

https://www.scriptslug.com/assets/scripts/breaking-bad-101-pilot-2008.pdf

https://www.scriptslug.com/assets/scripts/breaking-bad-301-no-mas-2010.pdf

https://www.scriptslug.com/assets/scripts/breaking-bad-306-sunset-2010.pdf

https://www.scriptslug.com/assets/scripts/breaking-bad-307-one-minute-2010.pdf

https://www.scriptslug.com/assets/scripts/breaking-bad-309-kafkaesque-2010.pdf

https://www.scriptslug.com/assets/scripts/breaking-bad-310-fly-2010.pdf

https://www.scriptslug.com/assets/scripts/breaking-bad-311-abiquiu-2010.pdf

https://www.scriptslug.com/assets/scripts/breaking-bad-312-half-measures-2010.pdf

Follow these steps to experiment on this dataset:

Prepare

```
python data/breaking_bad_char/prepare.py --num_scripts=1
```

Train

```
python train.py config/train_breaking_bad_char.py --device=mps --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=16 --n_head=16 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0 --ckpt_file=breaking_bad_ckpt.pt
```

Sample

```
python sample.py --out_dir=out-shakespeare-char --ckpt_file=breaking_bad_ckpt.pt --device=cpu
```

An interesting experiment I perform on this is to vary the number of characters in the input and obsereve the variation of the above evaluation metrics. You can very the input character size using the num_scripts flag in the prepare command above.

To produce the evalutation metrics result use the following commands :

BLEU score :

```
python evaluation_bleu.py --data_dir=breaking_bad_char
```

Spell Check :

```
python evaluation_spell_check.py --data_dir=breaking_bad_char
```

length of dataset in characters: 73,924 | BLEU Score: 0.4163732394366197 | Percentage of words not correctly spelled:  29.13732394366197 %

length of dataset in characters: 149,908 | BLEU Score: 0.4056261343012704 | Percentage of words not correctly spelled:  29.038112522686028 %

length of dataset in characters: 221,580 | BLEU Score: 0.3573407202216066 | Percentage of words not correctly spelled:  30.56325023084026 %

length of dataset in characters: 281,409 | BLEU Score: 0.39691189827429607 | Percentage of words not correctly spelled:  30.79019073569482 %

length of dataset in characters: 353,289 | BLEU Score: 0.3688888888888889 | Percentage of words not correctly spelled:  31.644444444444442 %

length of dataset in characters: 415,481 | BLEU Score: 0.3787740164684355 | Percentage of words not correctly spelled:  28.636779505946937 %

length of dataset in characters: 478,793 | BLEU Score: 0.39572192513368987 | Percentage of words not correctly spelled:  26.648841354723707 %

length of dataset in characters: 553,897 | BLEU Score: 0.4151291512915129 | Percentage of words not correctly spelled:  31.088560885608857 %

## Fine tuning

Fine tune training

```
python train.py config/finetune_breaking_bad.py --device=mps --compile=False
```

Comparing the pre-train and Fine tune data distributions

```
python compare_similarity.py
```

Results :

Data : 8 scripts | Training : 20 max_iters
BLEU Score for Shakespeare: 0.42727272727272725
BLEU Score for Breaking Bad: 0.3499999999999999

Data : 8 scripts | Training : 50 max_iters
BLEU Score for Shakespeare: 0.4200710479573712
BLEU Score for Breaking Bad: 0.3836589698046181

Data : 8 scripts | Training : 100 max_iters
BLEU Score for Shakespeare: 0.3698630136986301
BLEU Score for Breaking Bad: 0.37990867579908677




