# Novel-Object-Captioning
Novel Object Captioning: a repetition of Neural Baby Talk

First you need to configue the environment.
```
# download files
cd nbt
unzip data
unzip tools
unzip glove.6B -d .vector_cache

# environment configuration
apt-get update && \
    apt-get install -y \
    ant \
    vim \
    ca-certificates-java \
    nano \
    openjdk-8-jdk \
    unzip \
    wget && \
    apt-get clean
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 >> /etc/profile
source /etc/profile
update-ca-certificates -f && export JAVA_HOME

# configue the conda environment
conda env create -f environment.yml
conda activate ntt
pip install tensorflow==1.0.0 pyyaml
pip install bert_embedding -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Then you can run this to train the model!
```
python main.py --path_opt cfgs/noc_coco_res101.yml --batch_size 10 --cuda True --num_workers 8 --max_epoch 30 --checkpoint_path save/beam3
```
several important arguments:
- `path_opt` : you can choose another vgg model.
- `batch_size` : a bigger batch size will be better. I use this for my machine.
- `checkpoint_path` : where to save you record files.

Then you can evaluate the model. Remenber to change the file paths.
```
# get the all the metrics except F1
python eval.py 
# get F1 score
python F1.py
```
