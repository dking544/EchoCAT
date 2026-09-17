import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.dataloader import Mydataset, collate
from echocat.runtime import build, fixed_checkpoint, recipe, load_lines, CLASSES
from echocat.metrics import classification_metrics
from echocat.data import evaluation_dataset


def main():
    parser=argparse.ArgumentParser(description='Image inference/evaluation; model probabilities are softmaxed exactly once.')
    parser.add_argument('--task',required=True,choices=CLASSES)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image');group.add_argument('--list',dest='manifest')
    parser.add_argument('--checkpoint',help='Omit to use the bundled fixed checkpoint')
    parser.add_argument('--teacher')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--protocol',choices=['external','validation'],default='external',
        help='Original external PIL/torchvision pipeline or training-validation config pipeline')
    parser.add_argument('--legacy-extra-softmax',action='store_true',
        help='Six-class archived-image-score compatibility only; not a calibrated probability')
    a=parser.parse_args()
    if a.legacy_extra_softmax and a.task!='six':parser.error('Legacy score compatibility is six-class only')
    cfg=recipe(a.task)
    lines=load_lines(a.manifest,len(CLASSES[a.task])) if a.manifest else [f'{Path(a.image).resolve()} 0\n']
    dataset=evaluation_dataset(lines,a.task,a.protocol)
    loader=DataLoader(dataset,batch_size=a.batch_size,num_workers=a.workers,shuffle=False,collate_fn=collate)
    model=build(a.task,checkpoint=a.checkpoint or fixed_checkpoint(a.task),device=a.device,teacher=a.teacher).eval()
    result=[];all_scores=[];all_targets=[]
    with torch.no_grad():
        for images,targets,paths in loader:
            scores=model(images.to(a.device),return_loss=False)
            if a.legacy_extra_softmax:scores=scores.softmax(1)
            scores=scores.cpu().numpy();all_scores.append(scores);all_targets.append(targets.numpy())
            for path,score,label in zip(paths,scores,targets.tolist()):
                result.append({'image':str(path),'label':label if a.manifest else None,
                    'prediction':int(score.argmax()),'predicted_class':CLASSES[a.task][int(score.argmax())],
                    'scores':score.tolist()})
    output={'task':a.task,'class_names':CLASSES[a.task],'input_protocol':a.protocol,
            'score_protocol':'legacy_extra_softmax' if a.legacy_extra_softmax else 'single_softmax',
            'records':result}
    if a.manifest:output['metrics']=classification_metrics(np.concatenate(all_scores),np.concatenate(all_targets))
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(output.get('metrics',result),ensure_ascii=False))


if __name__=='__main__':main()
