CUDA_VISIBLE_DEVICES=1 python train.py \
--config configs/waymo_reconstruction.yaml \
source_path=data/waymo_scenes/0145050 \
model_path=eval_output/waymo_reconstruction/0145050_full

CUDA_VISIBLE_DEVICES=2 python separate.py \
--config configs/kitti_reconstruction.yaml \
source_path=data/kitti_mot/training/image_02/0001 \
model_path=eval_output/kitti_reconstruction/0001 \
start_frame=380 \
end_frame=431

CUDA_VISIBLE_DEVICES=2 python separate.py \
--config configs/kitti_nvs.yaml \
source_path=data/kitti_mot/training/image_02/0001 \
model_path=eval_output/kitti_nvs_aligned_poses/0001 \
scene_type=KittiDepth \
start_frame=380 \
end_frame=431