#!/bin/bash

audioset_train_all_json=~/workspace/datasets/ASVspoof/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.json
audioset_label=~/workspace/datasets/ASVspoof/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.label.csv
pretrained_model=~/workspace/project/AudioMAE/ckpt/pretrained.pth

dataset=asvspoof2019

python main_pretrain.py \
--batch_size 16 \
--norm_pix_loss True \
--model mae_vit_base_patch16 \
--mask_ratio 0.5 \
--epochs 50 \
--warmup_epochs 2 \
--save_every_epoch 2 \
--blr 5e-6 --weight_decay 0.0001 \
--dataset $dataset \
--data_train $audioset_train_all_json \
--label_csv $audioset_label \
--device cuda:2 \
--roll_mag_aug True \
--decoder_mode 1 \
--finetune $pretrained_model \



