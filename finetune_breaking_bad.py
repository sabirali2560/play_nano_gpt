# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char'
eval_interval = 5 # keep frequent because we'll overfit
eval_iters = 40
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'breaking-bad-char'
wandb_run_name = 'mini-gpt'

dataset = 'breaking_bad_char'
init_from = 'shakespeare'
gradient_accumulation_steps = 32
batch_size = 1
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

max_iters = 100

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

#checkpoint file
ckpt_file = 'breaking_bad_ckpt.pt'

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

