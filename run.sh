#!/bin/bash


python ./trainer.py --domain "fiction-travel-government-slate-telephone" --lr "4e-5" --num_train_epochs "3" --warmup_ratio "0.01" --grad_accum_steps "2"

#0.01/0.1, 4/5e-5

#python ./tuner.py --domain "fiction"
#python ./tuner.py --domain "travel"
#python ./tuner.py --domain "government"
#python ./tuner.py --domain "slate"
#python ./tuner.py --domain "telephone"


# Run tuning for different domains
#python ./trainer.py --domain "fiction" --lr "3e-5" --num_train_epochs "3" --warmup_ratio "0.06" --grad_accum_steps "2"
#python ./trainer.py --domain "travel" --lr "3e-5" --num_train_epochs "3" --warmup_ratio "0.06" --grad_accum_steps "2"
#python ./trainer.py --domain "government" --lr "3e-5" --num_train_epochs "3" --warmup_ratio "0.06" --grad_accum_steps "2"
#python ./trainer.py --domain "slate" --lr "3e-5" --num_train_epochs "3" --warmup_ratio "0.06" --grad_accum_steps "2"
#python ./trainer.py --domain "telephone" --lr "3e-5" --num_train_epochs "3" --warmup_ratio "0.06" --grad_accum_steps "2"

