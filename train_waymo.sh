# 现在尝试20
CUDA_VISIBLE_DEVICES=3 python train.py \
--config configs/waymo_nvs.yaml \
source_path=data/waymo_scenes/0017085 \
lambda_lidar=0 \
model_path=eval_output/waymo_nvs/0017085/aligned_poses_random_pc_10
# aligned_poses+lidar
CUDA_VISIBLE_DEVICES=2 python train.py \
--config configs/waymo_nvs.yaml \
source_path=data/waymo_scenes/0017085 \
lambda_lidar=0 \
model_path=eval_output/waymo_nvs/0017085/aligned_poses_lidar

# 0145050 random
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/waymo_nvs.yaml \
source_path=data/waymo_scenes/0145050 \
lambda_lidar=0 \
model_path=eval_output/waymo_nvs/0145050/aligned_poses_random_pc_10
# 0147030 random
CUDA_VISIBLE_DEVICES=2 python train.py \
--config configs/waymo_nvs.yaml \
source_path=data/waymo_scenes/0147030 \
lambda_lidar=0 \
model_path=eval_output/waymo_nvs/0147030/aligned_poses_random_pc_10
# 0158150 random
CUDA_VISIBLE_DEVICES=1 python train.py \
--config configs/waymo_nvs.yaml \
source_path=data/waymo_scenes/0158150 \
lambda_lidar=0 \
model_path=eval_output/waymo_nvs/0158150/aligned_poses_random_pc_10


# evaluate
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
--config configs/waymo_nvs.yaml \
source_path=data/waymo_scenes/0017085 \
lambda_lidar=0 \
model_path=eval_output/waymo_nvs/0017085/aligned_poses_random_pc_10
CUDA_VISIBLE_DEVICES=1 python evaluate.py \
--config configs/waymo_nvs.yaml \
source_path=data/waymo_scenes/0145050 \
lambda_lidar=0 \
model_path=eval_output/waymo_nvs/0145050/aligned_poses_random_pc_10
CUDA_VISIBLE_DEVICES=2 python evaluate.py \
--config configs/waymo_nvs.yaml \
source_path=data/waymo_scenes/0147030 \
lambda_lidar=0 \
model_path=eval_output/waymo_nvs/0147030/aligned_poses_random_pc_10
CUDA_VISIBLE_DEVICES=3 python evaluate.py \
--config configs/waymo_nvs.yaml \
source_path=data/waymo_scenes/0158150 \
lambda_lidar=0 \
model_path=eval_output/waymo_nvs/0158150/aligned_poses_random_pc_10


# 0017085 dynamo depth
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/waymo_nvs.yaml \
source_path=data/waymo_scenes/0017085 \
lambda_lidar=0 \
model_path=eval_output/waymo_nvs/0017085/aligned_poses_dynamo_depth
# 0145050 dynamo depth
CUDA_VISIBLE_DEVICES=2 python train.py \
--config configs/waymo_nvs.yaml \
source_path=data/waymo_scenes/0145050 \
lambda_lidar=0 \
model_path=eval_output/waymo_nvs/0145050/aligned_poses_dynamo_depth
# 0147030 dynamo depth
CUDA_VISIBLE_DEVICES=2 python train.py \
--config configs/waymo_nvs.yaml \
source_path=data/waymo_scenes/0147030 \
lambda_lidar=0 \
model_path=eval_output/waymo_nvs/0147030/aligned_poses_dynamo_depth
# 0158150 dynamo depth
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/waymo_nvs.yaml \
source_path=data/waymo_scenes/0158150 \
lambda_lidar=0 \
model_path=eval_output/waymo_nvs/0158150/aligned_poses_dynamo_depth