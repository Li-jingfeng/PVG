# seperate
CUDA_VISIBLE_DEVICES=0 python separate.py \
--config configs/kitti_nvs.yaml \
source_path=data/kitti_mot/training/image_02/0002 \
model_path=eval_output/kitti_nvs/0002_debug \
start_frame=121 \
end_frame=221

model_path=eval_output/kitti_nvs/0002_debug

CUDA_VISIBLE_DEVICES=1 python train.py \
--config configs/waymo_reconstruction.yaml \
source_path=data/waymo_scenes/0145050 \
model_path=eval_output/waymo_reconstruction/0145050_full

# 只是更换内参矩阵的cx cy  结果发现loss还是nan，使用的与126前一个实验一样（没有使用transfor matrix，就是集中相机方向和分布的）
CUDA_VISIBLE_DEVICES=2 python train.py \
--config configs/kitti_nvs.yaml \
source_path=data/kitti_mot/training/image_02/0001 \
model_path=eval_output/kitti_nvs/0001_both_aligned_loss?_change_cxcy \
start_frame=380 \
end_frame=431

# 使用了transfor matrix 结果发现loss还是nan
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/kitti_nvs.yaml \
source_path=data/kitti_mot/training/image_02/0001 \
model_path=eval_output/kitti_nvs/0001_both_aligned_loss?_change_cxcy_transform \
start_frame=380 \
end_frame=431

# 使用了transfor matrix 换个序列
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/kitti_nvs.yaml \
source_path=data/kitti_mot/training/image_02/0001 \
model_path=eval_output/kitti_nvs/0001_both_aligned_loss?_change_cxcy_transform \
start_frame=100 \
end_frame=151

# 使用了transfor matrix 换个序列 lidar有和没有一样差，接下来换成lidar做gs初始化（这样也不行），全部给真值
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/kitti_nvs.yaml \
source_path=data/kitti_mot/training/image_02/0001 \
model_path=eval_output/kitti_nvs/0001_both_aligned_loss?_change_cxcy_transform \
lambda_lidar=0 \
start_frame=100 \
end_frame=131

# 测试0002序列有无这个问题 看起来没有问题
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/kitti_nvs.yaml \
source_path=data/kitti_mot/training/image_02/0002 \
model_path=eval_output/kitti_nvs/0002 \
start_frame=170 \
end_frame=221
# lambda_lidar=0 \

# 测试0006序列有无这个问题
CUDA_VISIBLE_DEVICES=2 python train.py \
--config configs/kitti_nvs.yaml \
source_path=data/kitti_mot/training/image_02/0006 \
model_path=eval_output/kitti_nvs/0006 \
start_frame=50 \
end_frame=101
# lambda_lidar=0 \

CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/kitti_nvs.yaml \
source_path=data/kitti_mot/training/image_02/0006 \
model_path=eval_output/kitti_nvs/0006/all_images_original \
start_frame=0 \
end_frame=269

# 测试0001 0002 0006序列变长之后对于结果的影响
# 0001怎么挑都不行，0002和0006都可以找到结果好的长序列
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/kitti_nvs.yaml \
source_path=data/kitti_mot/training/image_02/0001 \
model_path=eval_output/kitti_nvs/0001/long_seq \
start_frame=181 \
end_frame=381

# 这个跑完了
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/kitti_nvs.yaml \
source_path=data/kitti_mot/training/image_02/0002 \
model_path=eval_output/kitti_nvs/0002/long_seq \
start_frame=121 \
end_frame=221
# start_frame=70 \
# 跑完了
CUDA_VISIBLE_DEVICES=1 python train.py \
--config configs/kitti_nvs.yaml \
source_path=data/kitti_mot/training/image_02/0006 \
model_path=eval_output/kitti_nvs/0006/long_seq \
start_frame=100 \
end_frame=200

# 现在尝试没有对单目深度做align
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/kitti_nvs.yaml \
source_path=data/kitti_mot/training/image_02/0001 \
model_path=eval_output/kitti_nvs/0001_debug/ \
start_frame=281 \
end_frame=381
# 0002长序列
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/kitti_nvs.yaml \
source_path=data/kitti_mot/training/image_02/0002 \
model_path=eval_output/kitti_nvs/0002_debug_ \
start_frame=121 \
end_frame=221

CUDA_VISIBLE_DEVICES=3 python train.py \
--config configs/kitti_nvs.yaml \
source_path=data/kitti_mot/training/image_02/0006 \
model_path=eval_output/kitti_nvs/0006_debug \
start_frame=100 \
end_frame=200
# 0001推荐序列
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/kitti_nvs.yaml \
source_path=data/kitti_mot/training/image_02/0001 \
model_path=eval_output/kitti_nvs/0001_depthnotalign/ \
start_frame=380 \
end_frame=431

# 使用vo的pose和depthanything的深度图
CUDA_VISIBLE_DEVICES=1 python train.py \
--config configs/kitti_nvs.yaml \
source_path=data/kitti_mot/training/image_02/0002 \
model_path=eval_output/kitti_nvs/0002_debug_vo_poses_depthanything \
start_frame=121 \
end_frame=221