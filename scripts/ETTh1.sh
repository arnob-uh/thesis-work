# export PYTHONPATH=../
# CUDA_DEVICE_ORDER=PCI_BUS_ID \
# python3 ../experiments/SCINet.py \
#    --dataset_type="ETTh1" \
#    --split_type="popular" \
#    --norm_type="RevIN" \
#    --device="cuda:0" \
#    --batch_size=32 \
#    --horizon=1 \
#    --pred_len="96" \
#    --windows=96 \
#    --epochs=100   \
#    runs --seeds='[3]'


# export PYTHONPATH=../
# CUDA_DEVICE_ORDER=PCI_BUS_ID \
# python3 ../experiments/SCINet.py \
#    --dataset_type="ETTh1" \
#    --split_type="popular" \
#    --norm_type="SAN" \
#    --device="cuda:0" \
#    --batch_size=32 \
#    --horizon=1 \
#    --pred_len="96" \
#    --windows=96 \
#    --epochs=100   \
#    runs --seeds='[3]'

# export PYTHONPATH=../
# CUDA_DEVICE_ORDER=PCI_BUS_ID \
# python3 ../experiments/SCINet.py \
#    --dataset_type="ETTh1" \
#    --split_type="popular" \
#    --norm_type="DAIN" \
#    --device="cuda:0" \
#    --batch_size=32 \
#    --horizon=1 \
#    --pred_len="96" \
#    --windows=96 \
#    --epochs=100   \
#    runs --seeds='[3]'

export PYTHONPATH=../
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ../experiments/SCINet.py \
   --dataset_type="ETTh1" \
   --split_type="popular" \
   --norm_type="CN" \
   --device="cuda:0" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len="96" \
   --windows=96 \
   --epochs=100   \
   runs --seeds='[3]'