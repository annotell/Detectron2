from omegaconf import OmegaConf
import os
import copy
import pickle
import cv2

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
    MetadataCatalog,
    DatasetCatalog,
)

from fvcore.transforms.transform import (
    CropTransform,
)

from detectron2.data.build import (
    print_instances_class_histogram,
)

from detectron2.evaluation import COCOEvaluator
from detectron2.data import detection_utils as utils
import torch
import numpy as np

dataloader = OmegaConf.create()

cls_to_index = {
    "Car": 0,
    "TwoWheeler": 1,
    "HeavyDuty": 2,
    "Pedestrian": 3,
    "Unclear": 4,
}

class_names = ["Car", "TwoWheeler", "HeavyDuty", "Pedestrian", "Unclear"]


def get_dataset_dict(root_dir: str, d: str):
    with open(os.path.join(root_dir, f"{d}.pickle"), "rb") as f:
        dataset_dicts = pickle.load(f)
    for record in dataset_dicts:
        record["file_name"] = os.path.join(root_dir, record["file_name"])
        record["to_mask"] = []
        temp_annotations = []
        for obj in record["annotations"]:
            if obj["category_id"] == 4:
                record["to_mask"].append(obj)
            else:
                temp_annotations.append(obj)
        record["annotations"] = temp_annotations

    print_instances_class_histogram(dataset_dicts, class_names)
    return dataset_dicts


def get_dataset_dict_test(root_dir: str, names: list):
    with open(os.path.join(root_dir, f"{names[0]}.pickle"), "rb") as f:
        dataset_dicts = pickle.load(f)
    for record in dataset_dicts:
        record["file_name"] = os.path.join(root_dir, record["file_name"])
        record["to_mask"] = []
        temp_annotations = []
        for obj in record["annotations"]:
            bbox = obj["bbox"]
            if bbox[0] >= 1668 and bbox[1] >= 828 and bbox[2] <= 2183 and bbox[3] <= 1343:
                obj["bbox"] -= np.array([1668, 828, 1668, 828])
                temp_annotations.append(obj)
        record["annotations"] = temp_annotations

    np.random.seed(0)
    dataset_dicts = np.random.choice(dataset_dicts, 1000, replace=False)
    print_instances_class_histogram(dataset_dicts, class_names)
    return dataset_dicts


def mask_not_relevant_objects(image, data_to_mask):
    for data_point in data_to_mask:
        box = data_point["bbox"]
        image = cv2.rectangle(
            np.array(image), (box[2], box[3]), (box[0], box[1]), (0, 0, 0), -1
        )

    return image


def mapper_camera_training(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="RGB")
    image = mask_not_relevant_objects(image, dataset_dict["to_mask"])
    image_size = 512
    augment_list = [
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomApply(
            T.RandomBrightness(intensity_min=0.8, intensity_max=1.2), prob=0.20
        ),
        T.RandomApply(
            T.RandomContrast(intensity_min=0.8, intensity_max=1.2), prob=0.20
        ),
        T.RandomApply(
            T.RandomSaturation(intensity_min=0.8, intensity_max=1.2), prob=0.20
        ),
        T.RandomCrop(crop_type="relative_range", crop_size=[0.6, 0.6]),
        T.FixedSizeCrop(crop_size=(image_size, image_size), pad=True),
    ]

    image, transforms = T.apply_transform_gens(augment_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


def mapper_camera_test(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="RGB")
    image = mask_not_relevant_objects(image, dataset_dict["to_mask"])
    augment_list = [
        CropTransform(
            x0=1668,
            y0=828,
            w=512,
            h=512,
            orig_h=2168,
            orig_w=3848,
        ),
    ]
    image, transforms = T.apply_transform_gens(augment_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    dataset_dict["height"] = 512
    dataset_dict["width"] = 512
    return dataset_dict


dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_dataset_dict)(root_dir="${..train.dataset.root_dir}", d="validation"),
    mapper=mapper_camera_training,
    total_batch_size=6,
    num_workers=6,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_dataset_dict_test)(root_dir="${..test.dataset.root_dir}"),
    mapper=mapper_camera_test,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="training",
    output_dir="./output_eval/",
)

dataloader.train.dataset.root_dir = "../cosmos_data_2dod/"
dataloader.test.dataset.root_dir = "../cosmos_data_2dod/"
dataloader.test.dataset.names = ["training"]

d = "training"
DatasetCatalog.register(
    d, lambda d=d: get_dataset_dict_test(dataloader.test.dataset.root_dir, [d])
)
MetadataCatalog.get(d).set(
    thing_classes=["Car", "TwoWheeler", "HeavyDuty", "Pedestrian"]
)
