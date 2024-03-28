## This is my improved version of the RVC (Realtime-Voice-Cloning) trainer repo

### Main improvements:
- Clean code, uses wandb for logging and accelerate for training
- Modular structure, you can change HifiGAN-NSF to BigVGAN-NSF
- Better prior network, instead of custom code for MultiHeadAttention with relative attention scores uses Pytorch MHA with flash attention
- PPG and RVMPE computation
- Insights about dataset

### Quickstart
```
PYTHONPATH=..:. python infer/modules/train/preprocess.py --inp_root /mnt/harddrive/datasets/singing_voice/downloaded_youtube_audios/audios/ --sr 40000 --n_p 16 --exp_dir ../ai-covers-backend/src/rvc/logs/pretrain_1228/ --per 3.0 --name2id_save_path ../ai-covers-backend/src/rvc/logs/pretrain_1228/youtube_artists.json --start_idx 300000
PYTHONPATH=..:. python infer/modules/train/extract_feature_print.py --exp_dir ../ai-covers-backend/src/rvc/logs/pretrain_1228/
PYTHONPATH=..:. python infer/modules/train/extract_f0_rmvpe.py --exp_dir ../ai-covers-backend/src/rvc/logs/pretrain_1228/ --is_half
accelerate launch src/train.py --exp_dir logs/pretrain_0327_with_yt/ --save_dir logs/pretrain_0327_with_yt/weights/ --config logs/pretrain_0327_with_yt/config.json --data_root /home/george/ai-covers-backend/src/rvc/ --save_interval 5 --total_epoch 1000 --batch_size 24 --lr 0.00002 --lr_decay 0.99 --wandb_project_name pretrain_0327_with_yt --enable_opt_load 0
```
