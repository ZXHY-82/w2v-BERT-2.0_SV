# OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  \
# torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
# --tag vox2_ \
# --is_distributed true \
# --yaml conf/w2v-bert/s1.yaml

# OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  \
# torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
# --tag vox2_ \
# --is_distributed true \
# --yaml conf/w2v-bert/s2.yaml \
# --pretrain results/checkpoints/vox2_251005144134/merge_lora.pth

# OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  \
# torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
# --tag vox2_ \
# --is_distributed true \
# --yaml conf/w2v-bert/s3.yaml \
# --pretrain results/checkpoints/vox2_251005145628/ckpt_0002.pth