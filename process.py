import json

noc_object = ['bottle','bus',  'couch', 'microwave', 'pizza', 'racket', 'suitcase', 'zebra']
dataset= ['/root/Desktop/pj3/annotations_DCC_clean/captions_split_set_%s_val_test_novel2014.json'%item for item in noc_object]

save_dict = None

for idx in range(len(dataset)):
    
    with open(dataset[idx], 'r') as f:
        ann = json.load(f)

    if save_dict is None:
        save_dict = ann
    else:
        save_dict['annotations'].extend(ann['annotations'])
        save_dict['images'].extend(ann['images'])
print(save_dict.keys())
with open('/root/Desktop/pj3/annotations_DCC_clean/alltest.json', 'w') as f:
    json.dump(save_dict, f)