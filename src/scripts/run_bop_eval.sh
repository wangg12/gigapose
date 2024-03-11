python core/fine_point_matching/engine/bop_eval_utils.py \
        --script-path third_party/bop_toolkit/scripts/eval_pose_results_more.py \
        --targets_name test_targets_bop19.json \
        --error_types "vsd,mssd,mspd" \
        --split test \
        --dataset ycbv \
        --result_names gpose2023_ycbv-test_f1a3e861-761d-41ea-b585-e84f9fd40591.csv \
        --result_dir /home/gu/Documents/2024/bop23_vis/gpose/


python core/fine_point_matching/engine/bop_eval_utils.py \
        --script-path third_party/bop_toolkit/scripts/eval_pose_results_more.py \
        --targets_name test_targets_bop19.json \
        --error_types "vsd,mssd,mspd" \
        --split test \
        --dataset lmo \
        --result_names gpose2023_lmo-test_87ee2d1b-1af9-4f6e-a51b-64b9a606222e.csv \
        --result_dir /home/gu/Documents/2024/bop23_vis/gpose/

python core/fine_point_matching/engine/bop_eval_utils.py \
        --script-path third_party/bop_toolkit/scripts/eval_pose_results_more.py \
        --targets_name test_targets_bop19.json \
        --error_types "vsd,mssd,mspd" \
        --split test \
        --dataset tless \
        --result_names gpose2023_tless-test_f6f57fde-c17f-49a8-9859-3dad44d18212.csv \
        --result_dir /home/gu/Documents/2024/bop23_vis/gpose/


python core/fine_point_matching/engine/bop_eval_utils.py \
        --script-path third_party/bop_toolkit/scripts/eval_pose_results_more.py \
        --targets_name test_targets_bop19.json \
        --error_types "vsd,mssd,mspd" \
        --split test \
        --dataset ycbv \
        --result_names genflow-multihypo16_ycbv-test_e77b7247-29c7-46bb-b468-11da990ca422.csv \
        --result_dir /home/gu/Documents/2024/bop23_vis/genflow/

python core/fine_point_matching/engine/bop_eval_utils.py \
        --script-path third_party/bop_toolkit/scripts/eval_pose_results_more.py \
        --targets_name test_targets_bop19.json \
        --error_types "vsd,mssd,mspd" \
        --split test \
        --dataset lmo \
        --result_names genflow-multihypo16_lmo-test_bb86b4a3-17ca-4a7a-b5ef-6d9007c62224.csv \
        --result_dir /home/gu/Documents/2024/bop23_vis/genflow/

python core/fine_point_matching/engine/bop_eval_utils.py \
        --script-path third_party/bop_toolkit/scripts/eval_pose_results_more.py \
        --targets_name test_targets_bop19.json \
        --error_types "vsd,mssd,mspd" \
        --split test \
        --dataset tless \
        --result_names genflow-multihypo16_tless-test_1763da63-c69a-4926-a494-49b163c93ed3.csv \
        --result_dir /home/gu/Documents/2024/bop23_vis/genflow/
