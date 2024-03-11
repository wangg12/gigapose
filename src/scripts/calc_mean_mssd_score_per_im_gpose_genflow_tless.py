import os
import os.path as osp
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np


if __name__ == "__main__":
    gpose_err_dir = Path("/home/gu/Documents/2024/bop23_vis/gpose/gpose2023_tless-test_f6f57fde-c17f-49a8-9859-3dad44d18212/error=mssd_ntop=-1")
    genflow_err_dir = Path("/home/gu/Documents/2024/bop23_vis/genflow/genflow-multihypo16_tless-test_1763da63-c69a-4926-a494-49b163c93ed3/error=mssd_ntop=-1")

    test_scenes = list(range(1, 21))

    result_path = Path("/home/gu/Documents/2024/bop23_vis/tless_gpose_genflow_mssd_scores_no_same_inst.csv")

    # stat how many same insts in each im
    test_targets_path = Path("/home/gu/Storage/BOP_DATASETS/tless/test_targets_bop19.json")
    test_targets = json.loads(test_targets_path.read_bytes())
    im_same_inst_nums = {}
    for item in test_targets:
        scene_id = item["scene_id"]
        im_id = item["im_id"]
        key = f"{scene_id:06d}/{im_id:06d}"
        if key not in im_same_inst_nums:
            im_same_inst_nums[key] = item["inst_count"]
        else:
            im_same_inst_nums[key] = max(item["inst_count"], im_same_inst_nums[key])

    # ------------------------------
    results = {}

    for test_scene in tqdm(test_scenes):
        gpose_error_path = gpose_err_dir / f"errors_{test_scene:06d}.json"
        genflow_error_path = genflow_err_dir / f"errors_{test_scene:06d}.json"
        assert gpose_error_path.exists(), gpose_error_path
        assert genflow_error_path.exists(), genflow_error_path
        gpose_scene_errors = json.loads(gpose_error_path.read_bytes())
        genflow_scene_errors = json.loads(genflow_error_path.read_bytes())
        # sceneID/imID gpose_mssd genflow_mssd delta_mssd
        for gpose_item in gpose_scene_errors:
            im_id = gpose_item["im_id"]
            key = f"{test_scene:06d}/{im_id:06d}"
            if key not in results:
                results[key] = {}
            if "gpose_mssd" not in results[key]:
                results[key]["gpose_mssd"] = []
            results[key]["gpose_mssd"].append(gpose_item["score"])

        for genflow_item in genflow_scene_errors:
            im_id = genflow_item["im_id"]
            key = f"{test_scene:06d}/{im_id:06d}"
            if key not in results:
                results[key] = {}
            if "genflow_mssd" not in results[key]:
                results[key]["genflow_mssd"] = []
            results[key]["genflow_mssd"].append(genflow_item["score"])

    avg_results = {}
    for key in tqdm(results):
        if not ("genflow_mssd" in results[key] and "gpose_mssd" in results[key]):
            continue
        avg_results[key] = [0, 0, 0]
        avg_results[key][0] = np.mean(results[key]["gpose_mssd"]) if len(results[key]["gpose_mssd"]) > 0 else 0
        avg_results[key][1] = np.mean(results[key]["genflow_mssd"]) if len(results[key]["genflow_mssd"]) > 0 else 0
        avg_results[key][2] = avg_results[key][0] - avg_results[key][1]

    # sort by delta mssd
    keys = list(avg_results.keys())
    delta_errors = [avg_results[key][2] for key in avg_results]
    inds = np.argsort(delta_errors)[::-1]

    with open(result_path, "w") as fw:
        fw.write("key,gpose_mssd,genflow_mssd,gpose-genflow_mssd\n")
        for ind in inds:
            key = keys[ind]
            if im_same_inst_nums[key] > 1:
                continue
            fw.write(f"{key},{avg_results[key][0]},{avg_results[key][1]},{avg_results[key][2]}\n")
    print("results wrote to {}".format(result_path))
    # import ipdb; ipdb.set_trace()
