from pycocotools.coco import COCO
import pickle
import pandas as pd

def eval_f1(keyword,pred,gt):
    tp=0;fp=0;fn=0
    df1=pd.DataFrame(pred)
    df2=pd.DataFrame(gt)
    for index in range(len(df1)):
        id=df1.loc[index].image_id
        pred_cap=df1.loc[index].caption
        pred_positive=True if pred_cap.find(keyword)!= -1 else False
        temp=df2[df2.image_id==id]
        len1=len(temp)
        if len1 == 0:
            continue
        temp=temp[temp.caption.str.contains(keyword)]
        if pred_positive and len(temp)==len1: tp+=1
        if pred_positive and len(temp)!=len1: fp+=1
        if not pred_positive and len(temp)==len1: fn+=1
    
    print('-'*5,'keyword:',keyword,'-'*5)
    print('tp: %d, fp: %d, fn: %d, F1_score: %.3f'%(tp,fp,fn,tp/(tp+0.5*fp+0.5*fn)))
    print('-'*25)
    return tp, fp, fn

            
pred_file='save/preds_test.pkl'
annFile_template='/root/Desktop/pj3/annotations_DCC_clean/captions_split_set_%s_val_test_novel2014.json' 

# prediction
with open(pred_file, 'rb') as f:
    pred = pickle.load(f)
# ground_truth

group1=['bottle','bus','couch','microwave','pizza','racket','suitcase','zebra']
tp_sum = fp_sum = fn_sum = 0
for name in group1:
    annFile = annFile_template % (name)
    coco = COCO(annFile)
    gt = coco.dataset['annotations']
    tp, fp, fn = eval_f1(name,pred,gt)
    tp_sum += tp
    fp_sum += fp
    fn_sum += fn

print('tp: %d, fp: %d, fn: %d, F1_score: %.3f'%(tp_sum,fp_sum,fn_sum,tp_sum/(tp_sum+0.5*fp_sum+0.5*fn_sum)))