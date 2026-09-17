import argparse
from collections import defaultdict
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from utils.dataloader import Mydataset,collate
from echocat.runtime import recipe,build,fixed_checkpoint,CLASSES
from echocat.aggregation import aggregate
from echocat.metrics import classification_metrics
from echocat.data import evaluation_dataset


def main():
    p=argparse.ArgumentParser(description='Run view and diagnosis models on the same images, then pool by fetus ID.\n\nInput JSONL rows: {"image": "relative/or/absolute/image.jpg", "fetus_id": "ID"}.\nOptional fetus_label is a fetus-level reference label, not an image label.\n')
    p.add_argument('--task',required=True,choices=['binary','six'])
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--view-checkpoint');p.add_argument('--diagnosis-checkpoint');p.add_argument('--teacher')
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args();source=Path(a.manifest).resolve()
    rows=[json.loads(line) for line in source.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not rows:raise ValueError('Empty fetal manifest')
    lines=[];groups=defaultdict(list)
    for i,row in enumerate(rows):
        if not isinstance(row['fetus_id'],str) or not row['fetus_id']:raise ValueError('Every image needs a fetus_id')
        image=Path(row['image']);image=image if image.is_absolute() else source.parent/image
        if not image.is_file():raise FileNotFoundError(image)
        lines.append(f'{image.resolve()} 0\n');groups[row['fetus_id']].append(i)
    if len(set(lines))!=len(lines):raise ValueError('Duplicate images in fetal manifest')
    probabilities={}
    for task,checkpoint in [('view',a.view_checkpoint),(a.task,a.diagnosis_checkpoint)]:
        loader=DataLoader(evaluation_dataset(lines,task,'external'),
                          batch_size=a.batch_size,shuffle=False,collate_fn=collate)
        model=build(task,checkpoint=checkpoint or fixed_checkpoint(task),device=a.device,
                    teacher=a.teacher if task!='view' else None).eval()
        with torch.no_grad():probabilities[task]=torch.cat([model(x.to(a.device),return_loss=False).cpu() for x,_,_ in loader]).numpy()
        del model
    result=[]
    for group,indices in groups.items():
        pooled=aggregate(probabilities['view'][indices,0],probabilities[a.task][indices],a.task)
        pooled['fetus_id']=group
        pooled['selected_indices']=[indices[i] for i in pooled['selected_indices']]
        labels={rows[i].get('fetus_label') for i in indices}
        if len(labels)>1:raise ValueError(f'Inconsistent fetal reference labels in fetus {group}')
        label=next(iter(labels))
        if label is not None and (not isinstance(label,int) or not 0<=label<len(CLASSES[a.task])):
            raise ValueError('Invalid fetus_label')
        pooled['label']=label;result.append(pooled)
    out={'task':a.task,'aggregation':'4CH>=0.5; top ceil(half); fallback top1; binary equal mean at 0.5; six 4CH-weighted mean',
         'class_names':CLASSES[a.task],'records':result}
    if all(x['label'] is not None for x in result):
        out['metrics']=classification_metrics([x['scores'] for x in result],[x['label'] for x in result],
                                             [x['prediction'] for x in result])
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out.get('metrics',{'fetuses':len(result)})))


if __name__=='__main__':main()
