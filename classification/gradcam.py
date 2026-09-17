import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from echocat.runtime import build,recipe,fixed_checkpoint,CLASSES
from utils.dataloader import Mydataset
from echocat.data import evaluation_dataset


def explain(model,x,target=None):

    activation=[]
    def capture(_module,_inputs,out):
        if not torch.is_tensor(out) or out.ndim!=3:
            raise RuntimeError('Expected [B,tokens,channels] last-stage activation')
        activation.append(out)
    hook=model.backbone.stages[-1].register_forward_hook(capture)
    try:
        model.eval()
        logits=model(x.detach().requires_grad_(True),return_loss=False,softmax=False)
        probabilities=logits.detach().softmax(1)[0]
        if target is None:target=int(probabilities.argmax())
        if not 0<=target<len(probabilities):raise ValueError('Invalid CAM target class')
        a=activation[-1]
        g=torch.autograd.grad(logits[0,target],a)[0]
        v=torch.relu((a*g.mean(1,keepdim=True)).sum(2))
        side=math.isqrt(v.shape[1])
        if side*side!=v.shape[1]:raise ValueError('Non-square spatial token grid')
        heat=F.interpolate(v.reshape(1,1,side,side),(224,224),mode='bilinear',align_corners=False)[0,0].detach().cpu().numpy()
        mx=float(heat.max());heat=heat/mx if mx>1e-14 else np.zeros_like(heat)
        return probabilities.cpu().numpy(),heat,target
    finally:
        hook.remove()


def main():
    parser=argparse.ArgumentParser(description='Grad-CAM on the exact preprocessed input; Jet, fixed 50% alpha.')
    parser.add_argument('--task',required=True,choices=CLASSES)
    parser.add_argument('--image',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--checkpoint');parser.add_argument('--teacher')
    parser.add_argument('--target',type=int,help='Omit to explain the predicted class')
    parser.add_argument('--protocol',choices=['external','validation'],default='external')
    parser.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    import matplotlib
    x=evaluation_dataset([f'{Path(args.image).resolve()} 0\n'],args.task,args.protocol)[0][0]
    model=build(args.task,checkpoint=args.checkpoint or fixed_checkpoint(args.task),device=args.device,teacher=args.teacher)
    probabilities,heat,target=explain(model,x[None].to(args.device),args.target)
    mean=np.asarray([123.675,116.28,103.53]);std=np.asarray([58.395,57.12,57.375])

    rgb=np.rint(x.permute(1,2,0).numpy()*std+mean).clip(0,255).astype(np.uint8)
    colored=matplotlib.colormaps['jet'](heat)[...,:3]*255
    overlay=(rgb*.5+colored*.5).clip(0,255).astype(np.uint8)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    Image.fromarray(rgb).save(out/'input.png');Image.fromarray(overlay).save(out/'gradcam.png')
    np.save(out/'heatmap.npy',heat)
    (out/'scores.json').write_text(json.dumps({'probabilities':probabilities.tolist(),
        'target_class':target,'class_name':CLASSES[args.task][target],
        'input_protocol':args.protocol,'target_layer':'backbone.stages[-1]','alpha':.5,'colormap':'jet'},indent=2))


if __name__=='__main__':main()
