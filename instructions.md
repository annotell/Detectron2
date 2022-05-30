### to train multi-gpu
python -B training.py --num-gpus X

###Using docker - recommended 
Assuming CUDA is version 11.3 and installs pytorch 1.10.
Before building the docker image make sure that /etc/docker/daemon.json allows
docker to access cuda at build time. The file should be as follows. If not,
update it and run: sudo systemctl restart docker

`{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}`

Steps:

1. Clone repo: git clone git@github.com:annotell/Detectron2.git detectron2
2. Build image with:  docker build -t detectron2-docker -f detectron2/docker/Dockerfile .
3. If building image for GCR repo, run instead: docker build -t eu.gcr.io/annotell-com/detectron2:TAGNAME -f detectron2/docker/Dockerfile .
4. To push to GCR repo, run: docker push eu.gcr.io/annotell-com/detectron2:TAGNAME
5. docker run -p 8889:8889 --hostname localhost -it -d --gpus all  -v /mnt/bfd/luca/cosmos_data_2dod/:/root/data -v /mnt/bfd/luca/cosmos_data_2dod/output/:/root/output/ --ipc=host detectron2-docker 
6. docker exec -it container_id bash


/path/to/output/folder/ is where detectron2 will save logs and models outside the docker image
