export HCCL_IF_IP=141.61.39.181
export GLOO_SOCKET_IFNAME=enp48s3u1u1
export TP_SOCKET_IFNAME=enp48s3u1u1
export HCCL_SOCKET_IFNAME=enp48s3u1u1

export VLLM_USE_V1=1

python3 data_parallel.py \
    --model="/mnt/nfs/weight/dsv3_w8a8" \
    --dp-size=4 \
    --tp-size=4 \
    --node-size=1 \
    --node-rank=0 \
    --master-addr=141.61.39.181 \
    --master-port=13345 \
    --trust-remote-code