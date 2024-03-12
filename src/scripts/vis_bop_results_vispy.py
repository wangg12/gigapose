from pathlib import Path
from bop_toolkit_lib import inout, misc
from bop_toolkit_lib import renderer

import os
import sys
import logging
import time

from tqdm import tqdm
import trimesh
from PIL import Image
import cv2
import numpy as np
from scipy import spatial
from skimage.feature import canny
from skimage.morphology import binary_dilation

PROJ_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJ_DIR))

from src.megapose.datasets.scene_dataset import (
    ObservationInfos,
    CameraData,
    ObjectData,
    SceneObservation,
)
from src.megapose.lib3d.transform import Transform
from src.megapose.datasets.object_dataset import RigidObjectDataset, RigidObject
from src.megapose.utils.conversion import convert_scene_observation_to_panda3d
from src.utils.trimesh import create_mesh_from_points, load_mesh

from src.libVis.numpy import get_cmap


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
results = {
    "gpose": {
        "itodd": "gpose2023_itodd-test_2487a8e7-deaf-445c-a732-7229902c4745.csv",
        "hb": "gpose2023_hb-test_6da106fb-4a62-45b0-ab14-0118c75bc497.csv",
        "icbin": "gpose2023_icbin-test_a043bf3e-311a-40ac-916c-a893a81dd0e9.csv",
        "lmo": "gpose2023_lmo-test_87ee2d1b-1af9-4f6e-a51b-64b9a606222e.csv",
        "tless": "gpose2023_tless-test_f6f57fde-c17f-49a8-9859-3dad44d18212.csv",
        "ycbv": "gpose2023_ycbv-test_f1a3e861-761d-41ea-b585-e84f9fd40591.csv",
        "tudl": "gpose2023_tudl-test_e60be2c3-35a9-4a30-b563-83284be6672b.csv",
    },
    "genflow": {
        "itodd": "genflow-multihypo16_itodd-test_91722167-50d9-48ce-98cc-dc34d148d78c.csv",
        "hb": "genflow-multihypo16_hb-test_947cd992-6058-480b-80e6-01f09a4f431f.csv",
        "icbin": "genflow-multihypo16_icbin-test_5bcbb2fb-24a3-43df-982b-3fadc69bb727.csv",
        "lmo": "genflow-multihypo16_lmo-test_bb86b4a3-17ca-4a7a-b5ef-6d9007c62224.csv",
        "tless": "genflow-multihypo16_tless-test_1763da63-c69a-4926-a494-49b163c93ed3.csv",
        "ycbv": "genflow-multihypo16_ycbv-test_e77b7247-29c7-46bb-b468-11da990ca422.csv",
        "tudl": "genflow-multihypo16_tudl-test_833fb644-15bb-45fb-be38-4038fca1a36b.csv",
    },
}


def draw_contour(img_PIL, mask, color, to_pil=True):
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    img = np.array(img_PIL)
    img[edge, :] = color
    if to_pil:
        return Image.fromarray(img)
    else:
        return img


def mask_background(gray_img, color_img, masks, color=(255, 0, 0), contour=True):
    """
    Put the color in the gray image according to the mask and add the contour
    """
    if isinstance(gray_img, Image.Image):
        gray_img = np.array(gray_img)
    for mask in masks:
        gray_img[mask > 0, :] = color_img[mask > 0, :]
        if contour:
            gray_img = draw_contour(gray_img, mask, color=color, to_pil=False)
    return gray_img


def group_by_image_level(data, image_key="im_id"):
    # group the detections by scene_id and im_id
    data_per_image = {}
    for det in data:
        scene_id, im_id = int(det["scene_id"]), int(det[image_key])
        key = f"{scene_id:06d}_{im_id:06d}"
        if key not in data_per_image:
            data_per_image[key] = []
        data_per_image[key].append(det)
    return data_per_image


def create_mesh_from_points(vertices, triangles, save_path):
    import xatlas
    vmapping, indices, uvs = xatlas.parametrize(vertices, triangles)
    xatlas.export(save_path, vertices[vmapping], indices, uvs)
    return trimesh.load(save_path)


def compute_image(im, triangles, colors):
    # Specify (x,y) triangle vertices
    image_height = im.shape[0]
    a = triangles[0]
    b = triangles[1]
    c = triangles[2]

    # Specify colors
    red = colors[0]
    green = colors[1]
    blue = colors[2]

    # Make array of vertices
    # ax bx cx
    # ay by cy
    #  1  1  1
    triArr = np.asarray([a[0], b[0], c[0], a[1], b[1], c[1], 1, 1, 1]).reshape((3, 3))

    # Get bounding box of the triangle
    xleft = min(a[0], b[0], c[0])
    xright = max(a[0], b[0], c[0])
    ytop = min(a[1], b[1], c[1])
    ybottom = max(a[1], b[1], c[1])

    # Build np arrays of coordinates of the bounding box
    xs = range(xleft, xright)
    ys = range(ytop, ybottom)
    xv, yv = np.meshgrid(xs, ys)
    xv = xv.flatten()
    yv = yv.flatten()

    # Compute all least-squares /
    p = np.array([xv, yv, [1] * len(xv)])
    alphas, betas, gammas = np.linalg.lstsq(triArr, p, rcond=-1)[0]

    # Apply mask for pixels within the triangle only
    mask = (alphas >= 0) & (betas >= 0) & (gammas >= 0)
    alphas_m = alphas[mask]
    betas_m = betas[mask]
    gammas_m = gammas[mask]
    xv_m = xv[mask]
    yv_m = yv[mask]

    def mul(a, b):
        # Multiply two vectors into a matrix
        return np.asmatrix(b).T @ np.asmatrix(a)

    # Compute and assign colors
    colors = mul(red, alphas_m) + mul(green, betas_m) + mul(blue, gammas_m)
    try:
        im[(image_height - 1) - yv_m, xv_m] = colors
    except:
        pass
    return im


def paint_texture(texture_image, min_val=240, max_val=255):
    """Paint the texture image to remove the white spots used in DiffDOPE"""
    # Convert the texture image to grayscale
    gray_texture = cv2.cvtColor(texture_image, cv2.COLOR_BGR2GRAY)
    # Find the white spots in the grayscale image (you may need to adjust the threshold)
    mask = cv2.inRange(gray_texture, min_val, max_val)
    # Apply inpainting to fill in the white spots
    texture_image = cv2.inpaint(
        texture_image, mask, inpaintRadius=1, flags=cv2.INPAINT_NS
    )
    return texture_image


def get_texture_distance(colors, mesh, resolution):
    start_time = time.time()
    texture_image = (
        np.ones((resolution, resolution, 3), dtype=np.uint8) * 255
    )  # Initialize the texture image (resxres resolution)
    uvs = mesh.visual.uv
    for triangle in mesh.faces:
        # print(triangle)
        c1, c2, c3 = triangle
        uv1 = (
            int(uvs[c1][0] * resolution),
            int(uvs[c1][1] * resolution),
        )
        uv2 = (
            int(uvs[c2][0] * resolution),
            int(uvs[c2][1] * resolution),
        )
        uv3 = (
            int(uvs[c3][0] * resolution),
            int(uvs[c3][1] * resolution),
        )
        # fill in the colors
        c_filling = [
            (int(colors[c1][0]), int(colors[c1][1]), int(colors[c1][2])),
            (int(colors[c2][0]), int(colors[c2][1]), int(colors[c2][2])),
            (int(colors[c3][0]), int(colors[c3][1]), int(colors[c3][2])),
        ]
        texture_image = compute_image(texture_image, [uv1, uv2, uv3], c_filling)
    # logger.info("Time to compute texture distance: {}".format(time.time() - start_time))
    texture = paint_texture(texture_image)
    return Image.fromarray(texture)



def rotation2quaternion(rot):
    import pyrr
    m = pyrr.Matrix33(
        [
            [rot[0], rot[1], rot[2]],
            [rot[3], rot[4], rot[5]],
            [rot[6], rot[7], rot[8]],
        ]
    )
    return m.quaternion


def filter_estimates(gt, pred_estimates):
    filtered_estimates = []
    for test_instance in gt:
        obj_id = test_instance["obj_id"]
        obj_estimate = [
            estimate for estimate in pred_estimates if estimate["obj_id"] == obj_id
        ]
        if len(obj_estimate) > 0:
            # keep the highest score
            obj_estimate = max(obj_estimate, key=lambda x: x["score"])
            filtered_estimates.append(obj_estimate)
    # sort by object_id
    # filtered_estimates = sorted(filtered_estimates, key=lambda x: x["obj_id"])
    return filtered_estimates


def get_edge(mask, bw=3, out_channel=3):
    if len(mask.shape) > 2:
        channel = mask.shape[2]
    else:
        channel = 1
    if channel == 3:
        mask = mask[:, :, 0] != 0
    edges = np.zeros(mask.shape[:2])
    edges[:-bw, :] = np.logical_and(mask[:-bw, :] == 1, mask[bw:, :] == 0) + edges[:-bw, :]
    edges[bw:, :] = np.logical_and(mask[bw:, :] == 1, mask[:-bw, :] == 0) + edges[bw:, :]
    edges[:, :-bw] = np.logical_and(mask[:, :-bw] == 1, mask[:, bw:] == 0) + edges[:, :-bw]
    edges[:, bw:] = np.logical_and(mask[:, bw:] == 1, mask[:, :-bw] == 0) + edges[:, bw:]
    if out_channel == 3:
        edges = np.dstack((edges, edges, edges))
    return edges


def compute_distances(
    pts,
    obj_label,
    gt_pose,
    pred_pose,
    symmetry_obj_ids,
    max_distance=10,
):
    # distance = np.zeros(len(pts))
    # R @ p + t  -> p^T @ R^T + t^T
    points_gt = pts @ gt_pose[:3, :3].T + gt_pose[:3, 3].reshape(1, 3)
    points_pred = pts @ pred_pose[:3, :3].T + pred_pose[:3, 3].reshape(1, 3)

    distance = np.linalg.norm(points_gt - points_pred, axis=1)
    distance_symmetry = spatial.distance_matrix(points_gt, points_pred, p=2).min(axis=1)

    obj_id = int(obj_label)
    if obj_id in symmetry_obj_ids or len(symmetry_obj_ids) == 0:
        distance = np.append(distance_symmetry, [0, max_distance])
    else:
        distance = np.append(distance, [max_distance])
    distance /= max_distance
    colors = get_cmap(distance, "turbo")
    return distance[:len(pts)], colors[:len(pts)]


if __name__ == "__main__":
    import imageio
    import copy
    from easydict import EasyDict as edict

    logging.basicConfig(level=logging.INFO)
    results_dir = Path("/home/gu/Documents/2024/bop23_vis/bop23_report")


    data_cfg = edict(
        root_dir=Path("/home/gu/Storage/BOP_DATASETS"),
        tless=dict(
            split="test_primesense",
            im_size=(540, 720),
        ),
        lmo=dict(
            split="test",
            im_size=(480, 640),
        ),
        ycbv=dict(
            split="test",
            im_size=(480, 640),
        )
    )

    selected_image_keys = {
        "lmo": [
            "000002_000217",
            "000002_000263",
            "000002_000283",
            "000002_000310",
            "000002_000402",
        ],
        "tudl": [
            "000001_001166",
            "000001_000457",
            "000003_001200",
            "000003_001400",
            "000003_001030",
        ],
        "ycbv": [
            # "000059_000242",
            # "000058_000940",
            "000057_001602",
            # "000049_000629",
            # "000051_000532",
        ],
        "tless": [
            "000007_000155",
            "000007_000285",
            "000002_000214",
            "000005_000070",
            "000001_000181",
            "000016_000161",
            "000016_000231",
        ],
    }

    for dataset_name in ["ycbv"]:  # "lmo", "tudl",
        if dataset_name == "ycbv":
            symmetry_obj_ids = [13, 18, 19, 20]
        elif dataset_name == "lmo":
            symmetry_obj_ids = [11, 12]
        else:
            symmetry_obj_ids = []
        save_dir = results_dir / "comparison" / dataset_name
        save_dir.mkdir(exist_ok=True, parents=True)

        # load target poses and group by image level
        target_path = results_dir / "target" / f"{dataset_name}.json"
        test_list = inout.load_json(target_path)
        test_list = group_by_image_level(test_list, image_key="im_id")

        # load estimates and group by image level
        estimates = {}
        for method in ["gpose", "genflow"]:
            path = results_dir / method / results[method][dataset_name]
            estimate = inout.load_bop_results(path)
            estimates[method] = group_by_image_level(estimate)

        test_data_cfg = data_cfg[dataset_name]
        split = test_data_cfg.split

        height, width = test_data_cfg.im_size
        ren = renderer.create_renderer(
            width, height, renderer_type="vispy", mode='rgb+depth', shading="flat")


        # initialize meshes
        bop_dir = data_cfg.root_dir
        dataset_dir = bop_dir / dataset_name

        cad_name = (
            "models" if dataset_name != "tless" else "models_reconst"
        )
        cad_dir = dataset_dir / cad_name
        model_infos = inout.load_json(cad_dir / "models_info.json")
        model_infos = [{"obj_id": int(obj_id)} for obj_id in model_infos.keys()]

        cad_eval_dir = dataset_dir / "models_eval"
        objects = []
        for model_info in tqdm(model_infos):
            obj_id = int(model_info["obj_id"])
            obj_label = f"obj_{obj_id:06d}"

            cad_path = (cad_dir / obj_label).with_suffix(".ply").as_posix()

            # Add object models.
            ren.add_object(obj_id, cad_path)

            object = RigidObject(
                label=str(model_info["obj_id"]),
                mesh_path=cad_path,
                mesh_units="mm",
                scaling_factor=1,
            )
            objects.append(object)

        mesh_vertices = {}
        ply_models = {}
        cad_path = {}
        for model_info in tqdm(model_infos):
            obj_id = int(model_info["obj_id"])
            obj_label = f"obj_{obj_id:06d}"
            cad_eval_path = (
                (cad_eval_dir / obj_label).with_suffix(".ply").as_posix()
            )
            cad_path[obj_label] = cad_eval_path

            ply_model = inout.load_ply(cad_eval_path)
            ply_models[obj_id] = ply_model
            mesh_vertices[obj_id] = np.array(ply_model["pts"])


        # image key that available in both estimates and target
        avail_keys = (
            set(estimates["genflow"].keys())
            & set(estimates["gpose"].keys())
            & set(test_list.keys())
        )

        for image_key in tqdm(avail_keys, desc="Rendering"):
            # try:
            if True:
                test_samples = test_list[image_key]
                # skip scene that have multiple instances of same object ID
                if len(set([obj["obj_id"] for obj in test_samples])) != len(test_samples):
                    continue

                scene_id, im_id = test_samples[0]["scene_id"], test_samples[0]["im_id"]
                if f"{scene_id:06d}_{im_id:06d}" not in selected_image_keys[dataset_name]:
                    continue
                img_path = dataset_dir / split / f"{scene_id:06d}/rgb/{im_id:06d}.png"
                img = Image.open(img_path)

                gt_info_path = dataset_dir / split / f"{scene_id:06d}/scene_gt_info.json"
                gt_info = inout.load_json(gt_info_path)[f"{im_id}"]
                gt_cam_path = dataset_dir / split / f"{scene_id:06d}/scene_camera.json"
                gt_cam = inout.load_json(gt_cam_path)[f"{im_id}"]
                gt_path = dataset_dir / split / f"{scene_id:06d}/scene_gt.json"
                gt = inout.load_json(gt_path)[f"{im_id}"]
                # keep only gt in test list

                gt = [
                    gt[i]
                    for i in range(len(gt))
                    if gt[i]["obj_id"] in [sample["obj_id"] for sample in test_samples]
                ]
                # sort objects by translation to camera
                gt = sorted(gt, key=lambda x: -np.linalg.norm(x["cam_t_m2c"]))
                scene_infos = ObservationInfos(scene_id=str(scene_id), view_id=str(im_id))
                pred_object_datas = {}
                count = {}
                for method, estimate in estimates.items():
                    pred_object_datas[method] = []
                    pred_estimate = estimate[image_key]
                    pred_estimate = filter_estimates(gt, pred_estimate)  # keep top 1
                    import ipdb; ipdb.set_trace()
                    count[method] = len(pred_estimate) == len(gt)
                    for idx_obj in range(len(pred_estimate)):
                        if str(pred_estimate[idx_obj]["obj_id"]) == "20":
                            print(method, np.array(pred_estimate[idx_obj]["R"]).reshape(3, 3))
                        object_data = ObjectData(
                            label=str(pred_estimate[idx_obj]["obj_id"]),
                            TWO=Transform(
                                np.array(pred_estimate[idx_obj]["R"]).reshape(3, 3),
                                np.array(pred_estimate[idx_obj]["t"]) * 0.001,
                            ),
                            unique_id=idx_obj,
                        )
                        pred_object_datas[method].append(object_data)
                if not (count["gpose"] and count["genflow"]):
                    continue

                object_datas = []
                for idx_obj in range(len(gt)):
                    object_data = ObjectData(
                        label=str(gt[idx_obj]["obj_id"]),
                        TWO=Transform(
                            np.array(gt[idx_obj]["cam_R_m2c"]).reshape(3, 3),
                            np.array(gt[idx_obj]["cam_t_m2c"]) * 0.001,
                        ),
                        unique_id=idx_obj,
                    )
                    object_datas.append(object_data)
                K = np.array(gt_cam["cam_K"]).reshape(3, 3)
                camera_data = CameraData(
                    K=K,
                    TWC=Transform(
                        np.eye(3),
                        np.zeros(3),
                    ),
                    resolution=np.array(img).shape[:2],
                )

                scene_obs = SceneObservation(
                    rgb=np.array(img),
                    object_datas=object_datas,
                    camera_data=camera_data,
                    infos=scene_infos,
                )

                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

                vis_imgs = [np.array(img)]
                save_path = save_dir / f"{image_key}"
                save_path.mkdir(exist_ok=True, parents=True)
                img.save(save_path / f"{image_key}_rgb.png")

                gray = cv2.cvtColor(np.array(img).copy(), cv2.COLOR_RGB2GRAY)
                gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

                ren_rgb_gt = gray.copy()
                # print("gt", [d_.label for d_ in object_datas])
                for idx_obj in range(len(object_datas)):
                    obj_label = object_datas[idx_obj].label

                    gt_obj_pose = copy.deepcopy(object_datas[idx_obj].TWO.toHomogeneousMatrix())
                    gt_obj_pose[:3, 3] *= 1000

                    obj_id = int(obj_label)

                    ren_res_gt = ren.render_object(
                        obj_id,
                        gt_obj_pose[:3, :3],
                        gt_obj_pose[:3, 3],
                        fx, fy, cx, cy)
                    ren_depth_gt = ren_res_gt["depth"]
                    # Convert depth image to distance image.
                    dist_gt = misc.depth_im_to_dist_im_fast(ren_depth_gt, K)
                    # Mask of the full object silhouette.
                    ren_mask_gt_i = dist_gt > 0
                    ren_rgb_gt_i = ren_res_gt["rgb"]
                    ren_rgb_gt[ren_mask_gt_i] = ren_rgb_gt_i[ren_mask_gt_i]
                    ren_edge_gt_i = get_edge(ren_mask_gt_i, out_channel=3)
                    ren_edge_gt_i[:, :, [0, 2]] = 0
                    ren_rgb_gt[ren_edge_gt_i > 0] = 255  # green
                overlay_imgs = [ren_rgb_gt]

                for method in ["gpose", "genflow"]:
                    ren_rgb_pred = gray.copy()
                    ren_heatmap_pred = np.ones((height, width, 3), dtype=np.uint8) * 255
                    # import ipdb; ipdb.set_trace()
                    # print(method, [d_.label for d_ in pred_object_datas[method]])
                    for idx_obj in range(len(pred_object_datas[method])):
                        obj_label = pred_object_datas[method][idx_obj].label

                        obj_pose = copy.deepcopy(pred_object_datas[method][
                            idx_obj
                        ].TWO.toHomogeneousMatrix())
                        obj_pose[:3, 3] *= 1000

                        gt_obj_pose = copy.deepcopy(object_datas[idx_obj].TWO.toHomogeneousMatrix())
                        gt_obj_pose[:3, 3] *= 1000
                        # if obj_label == "20" and dataset_name == "ycbv":
                        #     print("pred pose: ", obj_pose)
                        #     print("gt pose: ", gt_obj_pose)

                        obj_id = int(obj_label)
                        distances, dist_colors = compute_distances(
                            mesh_vertices[obj_id],
                            obj_label=obj_label,
                            gt_pose=gt_obj_pose,
                            pred_pose=obj_pose,
                            symmetry_obj_ids=symmetry_obj_ids,
                            max_distance=10,
                        )

                        ren_res_pred = ren.render_object(
                            obj_id,
                            obj_pose[:3, :3],
                            obj_pose[:3, 3],
                            fx, fy, cx, cy)
                        ren_depth_pred = ren_res_pred["depth"]
                        # Convert depth image to distance image.
                        dist_pred = misc.depth_im_to_dist_im_fast(ren_depth_pred, K)
                        # Mask of the full object silhouette.
                        ren_mask_pred_i = dist_pred > 0
                        ren_rgb_pred_i = ren_res_pred["rgb"]
                        ren_rgb_pred[ren_mask_pred_i] = ren_rgb_pred_i[ren_mask_pred_i]
                        ren_edge_pred_i = get_edge(ren_mask_pred_i, out_channel=3)
                        ren_edge_pred_i[:, :, [1, 2]] = 0
                        ren_rgb_pred[ren_edge_pred_i > 0] = 255  # red

                        heatmap_model = copy.deepcopy(ply_models[obj_id])
                        heatmap_model["colors"] = dist_colors.astype("float32")
                        # import ipdb; ipdb.set_trace()
                        heatmap_obj_id = obj_id + 1000
                        if heatmap_obj_id in ren.models:
                            ren.remove_object(heatmap_obj_id)
                        ren.add_ply_model(heatmap_obj_id, heatmap_model)
                        ren_res_heatmap = ren.render_object(
                            heatmap_obj_id,
                            gt_obj_pose[:3, :3],
                            gt_obj_pose[:3, 3],
                            fx, fy, cx, cy
                        )
                        ren_depth_heatmap = ren_res_heatmap["depth"]
                        # Convert depth image to distance image.
                        dist_heatmap = misc.depth_im_to_dist_im_fast(ren_depth_heatmap, K)
                        # Mask of the full object silhouette.
                        ren_mask_heatmap_i = dist_heatmap > 0
                        ren_rgb_heatmap_i = ren_res_heatmap["rgb"]
                        ren_heatmap_pred[ren_mask_heatmap_i] = ren_rgb_heatmap_i[ren_mask_heatmap_i]

                    overlay_imgs.append(ren_rgb_pred)
                    imageio.imwrite(f"{save_path}_overlay_{method}.png", ren_rgb_pred)

                    imageio.imwrite(f"{save_path}_heatmap_{method}.png", ren_heatmap_pred)
                    vis_imgs.append(ren_heatmap_pred)

                names = ["gt", "gpose", "genflow"]

                overlay_imgs = np.concatenate(overlay_imgs, axis=1)

                vis_imgs = np.concatenate(vis_imgs, axis=1)
                all_imgs = np.concatenate([vis_imgs, overlay_imgs], axis=0)

                all_imgs = Image.fromarray(all_imgs)
                all_imgs.save(save_dir / f"{image_key}.png")
            # except:
            #     pass
            print(save_dir / f"{image_key}.png")
