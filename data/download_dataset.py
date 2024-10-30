from google.cloud import bigquery
import pandas as pd
import json
import yaml
import numpy as np
import os
import glob
from tqdm import tqdm
from p_tqdm import p_umap
import pickle
from kognic.filestorage.filestorage import FileStorage
from kognic.filestorage.resource_parser import parse_file_id
import uuid
from io import BytesIO
from PIL import Image


def print_stats(t):
    print(f"\nNumber of images: {len(t['resource_id'])}")
    shape_classes = []
    for row in t["geometries"]:
        for geometry in row:
            geo = json.loads(geometry["shape_details"])
            shape_classes.append(geometry["shape_class"])
    shape_classes = np.array(shape_classes)
    counter = []
    classes = np.unique(shape_classes)
    scales = {c: [] for c in classes}
    coords = {c: [] for c in classes}
    for c in classes:
        mask = shape_classes == c
        counter.append(np.sum(mask))
    counter = np.array(counter)
    idx = np.argsort(-counter)
    counter = counter[idx]
    classes = classes[idx]
    print("\nClasses distribution:")
    for tot, cl in zip(counter, classes):
        print(f"{cl} --- tot: {tot:,}")


def create_sql_request(cfg):
    # Extract fields from config file
    table = cfg["table"]
    projects = cfg["projects"]
    requests = cfg["requests"]
    classes = cfg["classes"]
    if len(classes) > 1:
        classes = tuple(classes)
    else:
        classes = f"('{classes[0]}')"

    # Determine which ID list to use
    id_list = requests if requests else projects
    id_list_name = "request_id" if requests else "project_id"

    # Generate SQL query
    sql_query = f"""
    with 

    training as (
    SELECT * FROM {table}),
    meta as (select judgement_id, {id_list_name}, organization_id from annotell-com.dbt_staging__api_data.stg_api_data__judgement_overview group by 1,2,3),

    boxes as (
    select * from training left join meta using(judgement_id)
    )

    SELECT 
    input_internal_id, judgement_id, resource_id, ARRAY(SELECT AS STRUCT * FROM UNNEST(geometries) WHERE shape_class in {classes}) AS geometries, sensor_name
    FROM boxes
    WHERE boxes.judgement_id IN (SELECT last_judgement_id_in_chain FROM `annotell-com.dbt_assignment_chains.assignment_chains`)
    AND boxes.task_category = 'production'
    AND {id_list_name} in ({', '.join(map(str, id_list))})
    AND input_n_timestamps = 1
    """

    print(f"Selected {id_list_name}: {', '.join(map(str, id_list))}")
    return sql_query


def get_image(resource_id):
    filestorage_client = FileStorage()
    download_id = parse_file_id(resource_id)
    image_bytes = filestorage_client.get_file_content(download_id)
    image_pil = Image.open(BytesIO(image_bytes))

    return image_pil


def get_bbox_from_extreme(extreme_box):
    coords = extreme_box['coordinates']
    x_min = coords['minX']['coordinates'][0]
    x_max = coords['maxX']['coordinates'][0]
    y_min = coords['minY']['coordinates'][1]
    y_max = coords['maxY']['coordinates'][1]
    return [x_min, y_min, x_max, y_max]


def get_annotation(width, height, file_name, geometries):
    annotation = {}
    annotation["file_name"] = file_name
    annotation["image_id"] = str(uuid.uuid4())
    annotation["height"] = height
    annotation["width"] = width
    
    objs = []
    for feat in geometries:
        geo = json.loads(feat['shape_details'])
        box = get_bbox_from_extreme(geo)
        obj = {}
        obj["bbox_mode"] = 0  # BoxMode.XYXY_ABS
        obj["bbox"] = box
        obj["category_id"] = 0
        obj['object_type'] = feat['shape_class']
        objs.append(obj)
    annotation["annotations"] = objs
    return annotation


def download_single_input(item):
    judgement_id, resource_id, geometries, sensor_name, dataset_folder = item[1], item[2], item[3], item[4], item[5]

    filename = f"{judgement_id}-{sensor_name}"
    img_path = os.path.join(dataset_folder, 'images', f'{filename}.png')
    if os.path.exists(img_path):
        if os.path.exists(os.path.join(dataset_folder, 'annos', f'{filename}.pickle')):
            return True
    try:
        image = get_image(resource_id)
    except Exception as e:
        print(f"Failed to download image for {filename}")
        return False
    width, height = image.size
    annotation = get_annotation(width, height, img_path, geometries)
    image.save(os.path.join(dataset_folder, 'images', f"{filename}.png"))
    with open(os.path.join(dataset_folder, 'annos', f'{filename}.pickle'), 'wb') as handle:
        pickle.dump(annotation, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return True


def download_dataset(df, cfg):
    data = df.values.tolist()
    to_download = []
    for item in data:
        input_internal_id = item[0]
        judgement_id = item[1]
        sensor_name = item[4]
        filename = f"{judgement_id}-{sensor_name}"
        dataset_folder = os.path.join(cfg['dataset_root'], cfg['dataset_name'])
        os.makedirs(os.path.join(dataset_folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dataset_folder, 'annos'), exist_ok=True)
        item.append(dataset_folder)
        if os.path.exists(os.path.join(dataset_folder, 'images', f'{filename}.png')):
            if os.path.exists(os.path.join(dataset_folder, 'annos', f'{filename}.pickle')):
                continue
        to_download.append(item)
    
    nr_successful = sum(p_umap(download_single_input, to_download, num_cpus=8))
    print(f'Processed {nr_successful} inputs')
    
    
def create_splits(cfg):
    np.random.seed(42)
    all_annotations = []
    dataset_folder = os.path.join(cfg['dataset_root'], cfg['dataset_name'])
    annotation_files = np.random.permutation(glob.glob(os.path.join(dataset_folder, 'annos', '*.pickle')))
    
    for annotation_file in annotation_files:
        with open(annotation_file, 'rb') as handle:
            annotation = pickle.load(handle)
            all_annotations.append(annotation)
            
    split_idx = int(len(all_annotations) * cfg['train_split'])
    train_annotations = all_annotations[:split_idx]
    val_annotations = all_annotations[split_idx:]
    
    with open(os.path.join(dataset_folder, 'train.pickle'), 'wb') as f:
        pickle.dump(train_annotations, f)
    with open(os.path.join(dataset_folder, 'val.pickle'), 'wb') as f:
        pickle.dump(val_annotations, f)        
        

def copy_config(cfg):
    dataset_folder = os.path.join(cfg['dataset_root'], cfg['dataset_name'])
    config_path = os.path.join(dataset_folder, 'config.yaml')
    with open("config.yaml", 'r') as src, open(config_path, 'w') as dst:
        dst.write(src.read())
        
        
if __name__ == "__main__":
    # Read yaml file
    with open("config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    sql_query = create_sql_request(config)
    # Initialize BigQuery client
    client = bigquery.Client()
    # Run the query and convert to Pandas DataFrame
    print("Fetching data table...")
    df = client.query(sql_query).to_dataframe()
    # Remove rows where geometries is an empty list
    df = df[df['geometries'].map(lambda x: len(x) > 0)]
    print("Data table fetched successfully.")
    print_stats(df)
    # Ask user if they want to proceed with the download of images / annotations
    proceed = input("\nDo you want to proceed with the download of images? (y/n): ").strip().lower()
    if proceed == "y" or proceed == "":
        print("Proceeding with the download...")
        download_dataset(df, config)
        create_splits(config)
        # Copy config.yaml to the new dataset folder
        copy_config(config)
    else:
        print("Download aborted.")
