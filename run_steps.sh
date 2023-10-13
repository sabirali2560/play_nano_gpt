#!/bin/bash

for i in {1..8}
do

    mkdir output_num_scripts_$i
    
    # Prepare
    echo "Running Prepare... for num_scripts=$i"
    python data/breaking_bad_char/prepare.py --num_scripts=$i > output_num_scripts_$i/output_prepare.txt

    # Train
    echo "Running Training... for num_scripts=$i"
    python train.py config/train_breaking_bad_char.py --device=mps --compile=False --max_iters=500 > output_num_scripts_$i/output_train.txt

    # Sample
    echo "Running Sampling... for num_scripts=$i"
    python sample.py --out_dir=out-shakespeare-char --ckpt_file=breaking_bad_ckpt.pt --device=cpu > output_num_scripts_$i/output_sample.txt

    # Bleu score
    echo "Running Bleu evaluation.. for num_scripts=$i"
    python evaluation_bleu.py --data_dir=breaking_bad_char > output_num_scripts_$i/output_bleu.txt

    # Spell Check score
    echo "Running Spell Check evaluation.. for num_scripts=$i"
    python evaluation_spell_check.py --data_dir=breaking_bad_char > output_num_scripts_$i/output_spell_check.txt

done
