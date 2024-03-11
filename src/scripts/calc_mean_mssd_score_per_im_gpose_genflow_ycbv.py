import os
import os.path as osp
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np


if __name__ == "__main__":
    gpose_err_dir = Path("/home/gu/Documents/2024/bop23_vis/gpose/gpose2023_ycbv-test_f1a3e861-761d-41ea-b585-e84f9fd40591/error=mssd_ntop=-1")
    genflow_err_dir = Path("/home/gu/Documents/2024/bop23_vis/genflow/genflow-multihypo16_ycbv-test_e77b7247-29c7-46bb-b468-11da990ca422/error=mssd_ntop=-1")
    result_path = Path("/home/gu/Documents/2024/bop23_vis/ycbv_gpose_genflow_mmsd_scores.csv")

    results = {}

    test_scenes = [i for i in range(48, 59 + 1)]

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
            fw.write(f"{key},{avg_results[key][0]},{avg_results[key][1]},{avg_results[key][2]}\n")
    print("results wrote to {}".format(result_path))
    # import ipdb; ipdb.set_trace()
