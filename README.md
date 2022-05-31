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
conda activate nbt
pip install tensorflow==1.0.0 pyyaml
pip install bert_embedding -i https://pypi.tuna.tsinghua.edu.cn/simple

# preprocess the data
python process.py
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

Results:
|Method |BLEU1 |BLEU4 |CIDEr |METEOR |SPICE |F1|
|----|----|----|----|----|----|----|
|Greedy |69.5 |25.2 |69.0 |23.2 |16.7 |59.5|
|Beam Search| 69.7 |26.4 |69.5 |22.8 |16.8 |55.2|
|Paper |- |- |86.0 |24.1 |17.4 |70.3|

Eight objects:
|Metric |bottle |bus |couch |microwave |pizza |racket |suitcase |zebra|
|----|----|----|----|----|----|----|----|----|
|BLEU1 |74.2 |66.4 |75.4 |76.8 |67.5 |71.0| 65.3 |67.8|
|BLEU4 |30.4 |22.5| 32.7 |33.2 |25.0 |23.5 |25.5 |25.0|
|CIDEr |84.0 |47.5 |67.9 |55.4 |49.9| 31.5| 62.0 |42.6|
|METEOR |22.8 |21.5| 25.5| 24.4 |20.9 |24.6|20.8 |22.9|
|SPICE |16.2 |14.7| 18.5| 15.9 |16.4 |14.6 |13.1 |16.8|
|F1 |7.4 |77.2 |23.3| 42.1 |25.7 |8.5 |8.4| 73.4|
