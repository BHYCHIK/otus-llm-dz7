docker run --rm -it --gpus all -p 8888:8888 --shm-size=2g -v "$PWD":/workspace -v "$HOME"/otus/dz7:/notebooks llm-notebook
