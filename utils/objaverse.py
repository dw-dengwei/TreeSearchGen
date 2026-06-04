import os
import compress_json
import compress_pickle
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any

OBJATHOR_ASSETS_BASE_DIR = os.environ.get(
    "OBJATHOR_ASSETS_BASE_DIR", os.path.expanduser("~/.objathor-assets")
)


ASSETS_VERSION = os.environ.get("ASSETS_VERSION", "2023_09_23")
HD_BASE_VERSION = os.environ.get("HD_BASE_VERSION", "2023_09_23")
HOLODECK_BASE_DATA_DIR = os.path.join(
    OBJATHOR_ASSETS_BASE_DIR, "holodeck", HD_BASE_VERSION
)
OBJATHOR_VERSIONED_DIR = os.path.join(OBJATHOR_ASSETS_BASE_DIR, ASSETS_VERSION)
OBJATHOR_ASSETS_DIR = os.path.join(OBJATHOR_VERSIONED_DIR, "assets")
OBJATHOR_FEATURES_DIR = os.path.join(OBJATHOR_VERSIONED_DIR, "features")
OBJATHOR_ANNOTATIONS_PATH = os.path.join(OBJATHOR_VERSIONED_DIR, "annotations.json.gz")
HOLODECK_THOR_FEATURES_DIR = os.path.join(HOLODECK_BASE_DATA_DIR, "thor_object_data")
HOLODECK_THOR_ANNOTATIONS_PATH = os.path.join(
    HOLODECK_BASE_DATA_DIR, "thor_object_data", "annotations.json.gz"
)

if ASSETS_VERSION > "2023_09_23":
    THOR_COMMIT_ID = "8524eadda94df0ab2dbb2ef5a577e4d37c712897"
else:
    THOR_COMMIT_ID = "3213d486cd09bcbafce33561997355983bdf8d1a"

def get_asset_metadata(obj_data: Dict[str, Any]):
    if "assetMetadata" in obj_data:
        return obj_data["assetMetadata"]
    elif "thor_metadata" in obj_data:
        return obj_data["thor_metadata"]["assetMetadata"]
    else:
        raise ValueError("Can not find assetMetadata in obj_data")


def get_bbox_dims(obj_data: Dict[str, Any]):
    am = get_asset_metadata(obj_data)

    bbox_info = am["boundingBox"]

    if "x" in bbox_info:
        return bbox_info

    if "size" in bbox_info:
        return bbox_info["size"]

    mins = bbox_info["min"]
    maxs = bbox_info["max"]

    return {k: maxs[k] - mins[k] for k in ["x", "y", "z"]}

class ObjathorRetriever:
    def __init__(
        self,
        clip_model,
        clip_preprocess,
        clip_tokenizer,
        sbert_model,
        retrieval_threshold,
        device,
    ):
        self.device = device
        objathor_annotations = compress_json.load(OBJATHOR_ANNOTATIONS_PATH)
        # thor_annotations = compress_json.load(HOLODECK_THOR_ANNOTATIONS_PATH)
        # self.database = {**objathor_annotations, **thor_annotations}
        self.database = {**objathor_annotations}

        objathor_clip_features_dict = compress_pickle.load(
            os.path.join(OBJATHOR_FEATURES_DIR, "clip_features.pkl")
        )  # clip features
        objathor_sbert_features_dict = compress_pickle.load(
            os.path.join(OBJATHOR_FEATURES_DIR, "sbert_features.pkl")
        )  # sbert features
        assert (
            objathor_clip_features_dict["uids"] == objathor_sbert_features_dict["uids"]
        )

        objathor_uids = objathor_clip_features_dict["uids"]
        objathor_clip_features = objathor_clip_features_dict["img_features"].astype(
            np.float32
        )
        objathor_sbert_features = objathor_sbert_features_dict["text_features"].astype(
            np.float32
        )

        self.clip_features = torch.from_numpy(objathor_clip_features).to(device)
        self.clip_features = F.normalize(self.clip_features, p=2, dim=-1)

        self.sbert_features = torch.from_numpy(objathor_sbert_features).to(device)

        self.asset_ids = objathor_uids

        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.sbert_model = sbert_model


        self.use_text = True

    @staticmethod
    def compare_bbox_ratios(bbox1, bbox2, metric='cosine'):
        """
        比较两个Bounding Box的长宽高比例相似度。

        参数：
        - bbox1: 第一个Bounding Box的长宽高，格式为 [L, W, H]
        - bbox2: 第二个Bounding Box的长宽高，格式为 [L, W, H]
        - metric: 相似度度量方式，可选 'cosine', 'euclidean', 'relative'，默认为 'cosine'

        返回：
        - 相似度或距离值，根据选择的度量方式
        """
        # 计算比例向量
        def calculate_ratio_vector(bbox):
            norm = np.linalg.norm(bbox)  # 计算欧几里得长度
            return bbox / norm          # 标准化比例向量

        R1 = calculate_ratio_vector(bbox1)
        R2 = calculate_ratio_vector(bbox2)

        # 根据度量方式计算相似度或差异
        if metric == 'cosine':
            result = np.dot(R1, R2) / (np.linalg.norm(R1) * np.linalg.norm(R2))
        elif metric == 'euclidean':
            result = np.linalg.norm(R1 - R2)
        elif metric == 'relative':
            result = np.mean(np.abs((R1 - R2) / R1))
        else:
            raise ValueError("Invalid metric. Choose from 'cosine', 'euclidean', or 'relative'.")
        
        return result

    def retrieve(self, queries, target_size=None, threshold=28):
        device = self.device
        with torch.no_grad():
            query_feature_clip = self.clip_model.encode_text(
                self.clip_tokenizer(queries).to(device)
            )

            query_feature_clip = F.normalize(query_feature_clip, p=2, dim=-1)

        clip_similarities = 100 * torch.einsum(
            "ij, lkj -> ilk", query_feature_clip, self.clip_features
        )
        clip_similarities = torch.max(clip_similarities, dim=-1).values

        query_feature_sbert = self.sbert_model.encode(
            queries, convert_to_tensor=True, show_progress_bar=False
        )
        sbert_similarities = query_feature_sbert @ self.sbert_features.T

        if self.use_text:
            similarities = clip_similarities + sbert_similarities
        else:
            similarities = clip_similarities

        threshold_indices = torch.where(clip_similarities > threshold)

        unsorted_results = []
        for query_index, asset_index in zip(*threshold_indices):
            score = similarities[query_index, asset_index].item()
            if target_size is not None:
                size = get_bbox_dims(self.database[self.asset_ids[asset_index]])
                size = [size['x'], size['z'], size['y']]
                size_score = ObjathorRetriever.compare_bbox_ratios(
                    target_size[:2], size[:2]
                )
                score += 10 * size_score
            unsorted_results.append((self.asset_ids[asset_index], score))

        # Sorting the results in descending order by score
        results = sorted(unsorted_results, key=lambda x: x[1], reverse=True)

        return results

    def compute_size_difference(self, target_size, candidates):
        candidate_sizes = []
        for uid, _ in candidates:
            size = get_bbox_dims(self.database[uid])
            size_list = [size["x"] * 100, size["y"] * 100, size["z"] * 100]
            size_list.sort()
            candidate_sizes.append(size_list)

        candidate_sizes = torch.tensor(candidate_sizes)

        target_size_list = list(target_size)
        target_size_list.sort()
        target_size = torch.tensor(target_size_list)

        size_difference = abs(candidate_sizes - target_size).mean(axis=1) / 100
        size_difference = size_difference.tolist()

        candidates_with_size_difference = []
        for i, (uid, score) in enumerate(candidates):
            candidates_with_size_difference.append(
                (uid, score - size_difference[i] * 1)
            )

        # sort the candidates by score
        candidates_with_size_difference = sorted(
            candidates_with_size_difference, key=lambda x: x[1], reverse=True
        )

        return candidates_with_size_difference