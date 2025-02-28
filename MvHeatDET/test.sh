# train on multi-gpu
# NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
# torchrun --nproc_per_node=2 tools/train.py -c configs/evheat/MvHeatDET.yml \
#                         -r ckp/mvheatdet_input640_layers18_dim768.pth \
#                         --test-only


# test on single-gpu 
CUDA_VISIBLE_DEVICES=1 \
python tools/train.py -c configs/evheat/MvHeatDET.yml \
                        -r ckp/mvheatdet_input640_layers18_dim768.pth \
                        --test-only