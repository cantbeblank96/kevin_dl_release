conda activate pytorch2.0_gpu

cd /home/SENSETIME/xukaiming/Desktop/gitlab_repos/kevin_dl/
cur=`pwd`
export PYTHONPATH=$cur:$PYTHONPATH

python kevin_dl/tools/ckpts_management/scripts/only_keep_specific_ckpt.py \
--input_dir "/home/mnt/xukaiming/repos/kevin_dl/result/sombrero/jacobian_self_cancellation" \
--b_moved_to_bak 1 \
--b_dry_run 0 \
--argmax_metric ":test:acc_top_1" \
--specific_epoch "0" "10" "20" "30" "50" "100" "150" "200"


python kevin_dl/tools/ckpts_management/scripts/only_keep_specific_ckpt.py \
--input_dir "/home/mnt/xukaiming/repos/kevin_dl/result/sombrero/sombrero_v6p0b_over_resnet18_over_cifa100" \
--b_moved_to_bak 0 \
--b_dry_run 0 \
--argmax_metric ":test:acc_top_1" \
--specific_epoch "200"


python kevin_dl/tools/ckpts_management/scripts/only_keep_specific_ckpt.py \
--input_dir "/home/mnt/xukaiming/repos/kevin_dl/result/sombrero/sombrero_v6p7" \
--b_moved_to_bak 1 \
--b_dry_run 0