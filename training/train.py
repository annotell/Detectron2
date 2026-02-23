import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

import copy
import glob
import logging
import numpy as np
import os
import pickle
import uuid
from collections import OrderedDict
from PIL import Image

logger = logging.getLogger("detectron2")


def build_coco_dataset(data_root, class_map, split_ratio=0.98, cache_dir=None):
    """Build COCO-format dataset dicts from raw 2D annotations.

    Scans {data_root}/images/ for .webp files, matches to
    {data_root}/annos/{stem}_ExtremePointBox.pickle annotations,
    and converts to Detectron2 COCO format.

    Caches result as train_coco.pickle / val_coco.pickle in cache_dir
    (defaults to data_root if not specified).

    Args:
        data_root: Path to dataset directory (contains images/ and annos/)
        class_map: Dict {data_class: model_class}
        split_ratio: Train/val split ratio (default 0.98)
        cache_dir: Directory to write cache files (must be writable).
                   Falls back to data_root if None.

    Returns:
        (train_dicts, val_dicts): Lists of COCO-format dataset dicts
    """
    # Check both cache_dir and data_root for cached splits
    cache_base = cache_dir or data_root
    train_path = os.path.join(cache_base, "train_coco.pickle")
    val_path = os.path.join(cache_base, "val_coco.pickle")
    # Also check data_root (GCS FUSE may have pre-built cache from orchestrator)
    train_path_alt = os.path.join(data_root, "train_coco.pickle")
    val_path_alt = os.path.join(data_root, "val_coco.pickle")

    # Check cache_dir first, then data_root
    for tp, vp in [(train_path, val_path), (train_path_alt, val_path_alt)]:
        if os.path.exists(tp) and os.path.exists(vp):
            logger.info(f"Loading cached COCO splits from {os.path.dirname(tp)}")
            with open(tp, "rb") as f:
                train_dicts = pickle.load(f)
            with open(vp, "rb") as f:
                val_dicts = pickle.load(f)
            logger.info(f"  Train: {len(train_dicts)} images, Val: {len(val_dicts)} images")
            return train_dicts, val_dicts

    # Derive category_ids from class_map (sorted unique model classes)
    model_classes = sorted(set(class_map.values()))
    class_to_cat_id = {mc: i for i, mc in enumerate(model_classes)}

    logger.info(f"Building COCO dataset from {data_root}")
    logger.info(f"  Class mapping: {class_map}")
    logger.info(f"  Model classes -> category_id: {class_to_cat_id}")

    images_dir = os.path.join(data_root, "images")
    annos_dir = os.path.join(data_root, "annos")

    records = []
    skipped_no_anno = 0
    skipped_no_valid = 0

    # Cache annotation files to avoid re-reading for each camera view
    anno_cache = {}

    for img_file in sorted(glob.glob(os.path.join(images_dir, "*.webp"))):
        stem = os.path.splitext(os.path.basename(img_file))[0]

        # Try exact match first: {stem}_ExtremePointBox.pickle
        anno_file = os.path.join(annos_dir, f"{stem}_ExtremePointBox.pickle")
        if not os.path.exists(anno_file):
            # Fallback: annotations may be per-timestamp (sensor=None),
            # e.g. {judgement_id}_{timestamp}_None_ExtremePointBox.pickle
            parts = stem.split("_")
            if len(parts) >= 2:
                base_id = f"{parts[0]}_{parts[1]}"
                anno_file = os.path.join(annos_dir, f"{base_id}_None_ExtremePointBox.pickle")

        if not os.path.exists(anno_file):
            skipped_no_anno += 1
            continue

        try:
            with Image.open(img_file) as img:
                width, height = img.size
        except Exception:
            continue  # Skip corrupted/empty images

        if anno_file not in anno_cache:
            with open(anno_file, "rb") as f:
                anno_cache[anno_file] = pickle.load(f)
        raw_annos = anno_cache[anno_file]

        coco_annos = []
        for anno in raw_annos:
            data_class = anno["class"]
            model_class = class_map.get(data_class)
            if model_class is None:
                continue  # Not in mapping -> ignored
            bbox = anno["bbox"]
            # Filter bboxes that don't fit this camera's image dimensions
            x_min, y_min, x_max, y_max = bbox
            if x_max > width or y_max > height or x_min < 0 or y_min < 0:
                continue
            coco_annos.append({
                "bbox_mode": 0,  # BoxMode.XYXY_ABS
                "bbox": bbox,
                "category_id": class_to_cat_id[model_class],
                "object_type": model_class,
            })

        if not coco_annos:
            skipped_no_valid += 1
            continue

        records.append({
            "file_name": os.path.abspath(img_file),
            "image_id": str(uuid.uuid4()),
            "height": height,
            "width": width,
            "annotations": coco_annos,
        })

    logger.info(f"  Total images with annotations: {len(records)}")
    if skipped_no_anno > 0:
        logger.info(f"  Skipped (no annotation file): {skipped_no_anno}")
    if skipped_no_valid > 0:
        logger.info(f"  Skipped (no valid classes): {skipped_no_valid}")

    if len(records) == 0:
        raise ValueError(
            f"No valid image-annotation pairs found in {data_root}. "
            f"Expected images in {images_dir} and ExtremePointBox annotations in {annos_dir}."
        )

    # Shuffle and split
    rng = np.random.RandomState(42)
    rng.shuffle(records)
    split_idx = int(len(records) * split_ratio)
    train_dicts, val_dicts = records[:split_idx], records[split_idx:]

    logger.info(f"  Train: {len(train_dicts)} images, Val: {len(val_dicts)} images")

    # Cache to writable directory
    try:
        os.makedirs(cache_base, exist_ok=True)
        with open(train_path, "wb") as f:
            pickle.dump(train_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(val_path, "wb") as f:
            pickle.dump(val_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"  Cached COCO splits to {cache_base}")
    except OSError:
        logger.warning(f"  Could not cache COCO splits to {cache_base} (read-only?)")

    return train_dicts, val_dicts


def mapper_camera_training(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="RGB")
    augment_list = [
        T.RandomApply(T.RandomBrightness(intensity_min=0.9, intensity_max=1.1), prob=0.1),
        T.RandomApply(T.RandomContrast(intensity_min=0.9, intensity_max=1.1), prob=0.1),
        T.ResizeShortestEdge([800, 1080], 2000, "choice"),
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
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="RGB")

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    annos = [obj for obj in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict


def get_evaluator(cfg, dataset_name, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper_camera_test)
        evaluator = get_evaluator(cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name))
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter, max_to_keep=1)

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    data_loader = build_detection_train_loader(cfg, mapper=mapper_camera_training)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if cfg.TEST.EVAL_PERIOD > 0 and iteration % cfg.TEST.EVAL_PERIOD == 0 and iteration != max_iter:
                do_test(cfg, model)
                comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args, num_classes):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_list(args.opts)
    cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.DATASETS.TRAIN = ("autobaans_train",)
    cfg.DATASETS.TEST = ("autobaans_val",)
    cfg.OUTPUT_DIR = args.output_dir
    cfg.TEST.EVAL_PERIOD = 0
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size * args.num_gpus
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = [int(args.max_iter * 0.6), int(args.max_iter * 0.8)]
    cfg.SOLVER.GAMMA = 0.1
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.MODEL.BACKBONE.FREEZE_AT = 4

    # Auto-scale LR for distributed training (linear scaling rule).
    # When running on more GPUs than the config was tuned for, scale LR proportionally.
    BASE_GPUS = int(os.environ.get("BASE_GPUS", "2"))
    actual_gpus = args.num_gpus * args.num_machines
    if actual_gpus > BASE_GPUS and os.environ.get("TRAINING_MODE") == "gcp":
        scale_factor = actual_gpus / BASE_GPUS
        original_lr = cfg.SOLVER.BASE_LR
        cfg.SOLVER.BASE_LR = original_lr * scale_factor
        cfg.SOLVER.WARMUP_ITERS = max(cfg.SOLVER.WARMUP_ITERS, 1000)
        logger.info(
            f"Auto-scaled LR: {original_lr} -> {cfg.SOLVER.BASE_LR} "
            f"(scale={scale_factor:.1f}x, {BASE_GPUS} -> {actual_gpus} GPUs), "
            f"warmup_iters={cfg.SOLVER.WARMUP_ITERS}"
        )

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    # Configure logging early so build_coco_dataset messages are visible
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    # Parse class mapping from CLI: "Car:Vehicle,Person:Pedestrian"
    class_map = dict(pair.split(":") for pair in args.class_map.split(","))
    model_classes = sorted(set(class_map.values()))

    # Build COCO dataset from raw annotations.
    # Use output_dir for cache (data_root may be read-only GCS FUSE mount).
    os.makedirs(args.output_dir, exist_ok=True)
    train_dicts, val_dicts = build_coco_dataset(
        args.data_root, class_map, cache_dir=args.output_dir
    )

    cfg = setup(args, num_classes=len(model_classes))

    DatasetCatalog.register("autobaans_train", lambda: train_dicts)
    DatasetCatalog.register("autobaans_val", lambda: val_dicts)
    MetadataCatalog.get("autobaans_train").set(thing_classes=model_classes)
    MetadataCatalog.get("autobaans_val").set(thing_classes=model_classes)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(model, device_ids=[comm.get_local_rank()], broadcast_buffers=False)

    do_train(cfg, model)
    return do_test(cfg, model)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--data-root", type=str, required=True,
                        help="Dataset directory (contains images/ and annos/)")
    parser.add_argument("--class-map", type=str, required=True,
                        help="Class mapping as data:model pairs, e.g. 'Car:Vehicle,Person:Pedestrian'")
    # --output-dir is already provided by default_argument_parser()
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_root, "output_2d")

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
