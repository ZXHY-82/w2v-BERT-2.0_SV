# OMP_NUM_THREADS="12" CUDA_VISIBLE_DEVICES="0,1,2,3"  \
# torchrun --nnodes 1 --nproc_per_node=4 --master_port=12885 train_prune_s1.py \
# --tag prune_ \
# --is_distributed true \
# --yaml conf/prune/dis_prune_s1.yaml


# OMP_NUM_THREADS="12" CUDA_VISIBLE_DEVICES="0,1,2,3"  \
# torchrun --nnodes 1 --nproc_per_node=4 --master_port=12885 train_prune_s2.py \
# --tag prune_ \
# --is_distributed true \
# --yaml conf/prune/dis_prune_s2.yaml \
# --pretrain results/checkpoints/prune_s1/prune_update.pth


OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3"  \
torchrun --nnodes 1 --nproc_per_node=4 --master_port=12886 train.py \
--tag prune_ft_ \
--is_distributed true \
--yaml conf/prune/s1.yaml \
--pretrain results/checkpoints/prune_s2/prune_dis.pth

# OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3"  \
# torchrun --nnodes 1 --nproc_per_node=4 --master_port=12886 train.py \
# --tag prune_ft_ \
# --is_distributed true \
# --yaml conf/prune/s2.yaml \
# --pretrain results/checkpoints/xxx

# OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3"  \
# torchrun --nnodes 1 --nproc_per_node=4 --master_port=12886 train.py \
# --tag prune_ft_ \
# --is_distributed true \
# --yaml conf/prune/s3.yaml \
# --pretrain results/checkpoints/xxx
