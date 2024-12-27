#!/bin/bash

audioset_train_all_json=~/workspace/datasets/ASVspoof/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.json
audioset_label=~/workspace/datasets/ASVspoof/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.label.csv
pretrained_model=~/workspace/project/AudioMAE/ckpt/pretrained.pth

dataset=audioset

python main_pretrain.py \
--batch_size 8 \
--norm_pix_loss True \
--model mae_vit_base_patch16 \
--mask_ratio 0.8 \
--epochs 33 \
--warmup_epochs 3 \
--save_every_epoch 4 \
--blr 2e-4 --weight_decay 0.0001 \
--dataset $dataset \
--data_train $audioset_train_all_json \
--label_csv $audioset_label \
--device cuda:2 \
--roll_mag_aug True \
--decoder_mode 1 \
--finetune $pretrained_model \



