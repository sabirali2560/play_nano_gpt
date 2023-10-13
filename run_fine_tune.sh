#!/bin/bash

num_scripts=(1 2 4 8)
for i in "${num_scripts[@]}";
do

    mkdir output_fine_tune_$i
        
    # Prepare
    echo "Running Prepare... for num_scripts=$i"
    python data/breaking_bad_char/prepare_fine_tune.py --num_scripts=$i > output_fine_tune_$i/output_prepare.txt

    max_iters=(20 50 100)

    for j in "${max_iters[@]}";
    do

        # Train
        echo "Running Training... for num_scripts=$i max_iters=$j"
        python train.py config/finetune_breaking_bad.py --device=mps --compile=False --max_iters=$j > output_fine_tune_$i/output_train_$j.txt

        # Sample
        echo "Running Sampling... for num_scripts=$i max_iters=$j"
        python sample.py --out_dir=out-shakespeare-char --ckpt_file=breaking_bad_ckpt.pt --device=cpu > output_fine_tune_$i/output_sample_$j.txt

        # Comparison
        echo "Running Comparison.. for num_scripts=$i max_iters=$j"
        python compare_similarity.py > output_fine_tune_$i/output_comparison_$j.txt
    
    done

done
