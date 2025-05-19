# train on multi-gpu
# NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
# torchrun --nproc_per_node=2 tools/train.py -c configs/evheat/cvheat.yml -r path_to_ckp --test-only


# training on single-gpu 
CUDA_VISIBLE_DEVICES=0 \
python tools/train.py -c configs/evheat/cvheat.yml \
                        -r path_to_ckp \
                        --test-only