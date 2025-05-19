# train on multi-gpu
# NCCL_P2P_DISABLE=1 \
# CUDA_VISIBLE_DEVICES=4,6 \
# torchrun --nproc_per_node=2 tools/train.py -c configs/evheat/cvheat.yml -o output/contour


# training on single-gpu 
CUDA_VISIBLE_DEVICES=4 \
python tools/train.py -c configs/evheat/cvheat.yml -o output/contour