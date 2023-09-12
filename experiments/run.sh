cd /p/compressionleakage/dissecting_dist_inf/experiments
CUDA_VISIBLE_DEVICES=0 python train_models.py --load_config "$base_dir"/configs/celeba/train_models_compress.json --value 1.0 --split victim --num_models 20 --prop Eyeglasses
# CUDA_VISIBLE_DEVICES=0 python train_models.py --load_config configs/celeba/train_models_resnet_new.json --value 1.0 --split adv --num_models 40
# CUDA_VISIBLE_DEVICES=1 python blackbox_attacks.py --load_config /p/compressionleakage/dissecting_dist_inf/experiments/configs/celeba/blackbox_attacks.json --en compress_KL_attack_vic_15_shadow_30
# CUDA_VISIBLE_DEVICES=0 python blackbox_attacks.py --load_config /p/compressionleakage/dissecting_dist_inf/experiments/configs/celeba/blackbox_attacks_scratch.json --en compress_KL_attack_scratch_vic_15_shadow_30
# CUDA_VISIBLE_DEVICES=0 python whitebox_pin.py --load_config /p/compressionleakage/dissecting_dist_inf/experiments/configs/celeba/pin.json --en whitebox_attack
# CUDA_VISIBLE_DEVICES=1 python blackbox_attacks.py --load_config /p/compressionleakage/dissecting_dist_inf/experiments/configs/celeba/loss_attacks.json --en compress_loss_attack
# CUDA_VISIBLE_DEVICES=1 python blackbox_attacks.py --load_config /p/compressionleakage/dissecting_dist_inf/experiments/configs/celeba/loss_attacks_scratch.json --en compress_loss_attack_scratch