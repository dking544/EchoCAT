import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch
from echocat.runtime import build,recipe,fixed_checkpoint,load_lines,CLASSES,ROOT
from echocat.metrics import classification_metrics,selection_key
from echocat.aggregation import aggregate
from utils.dataloader import Mydataset
from utils.train_utils import set_random_seed


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_metrics_six_and_selection(self):
        scores=np.eye(6)[[0,1,2,3,4,0]]
        m=classification_metrics(scores,np.arange(6));m['avg_loss']=1.
        self.assertAlmostEqual(m['recall_macro'],5/6)
        self.assertEqual(m['f1_per_class'][5],0)
        self.assertEqual(selection_key(m,'six')[0],5/6)
        self.assertNotEqual(selection_key(m,'six','macro_f1')[0],m['accuracy'])
        absent=classification_metrics(np.eye(6)[:5],np.arange(5));absent['avg_loss']=1.
        with self.assertRaises(ValueError):selection_key(absent,'six')
        m=classification_metrics([[.9,.1],[.1,.9]],[0,1]);m['avg_loss']=.1
        self.assertEqual(selection_key(m,'binary')[0],1)

    def test_fetal_aggregation(self):
        r=aggregate([.9,.7,.5,.1],[[.9,.1],[.1,.9],[0,1],[0,1]],'binary')
        self.assertEqual(r['selected_indices'],[0,1]);self.assertEqual(r['prediction'],1)
        r=aggregate([.1,.2],[[1,0],[0,1]],'binary')
        self.assertTrue(r['used_fallback']);self.assertEqual(r['selected_indices'],[1])
        r=aggregate([0],[np.eye(6)[4]],'six');self.assertEqual(r['prediction'],4)
        self.assertEqual(aggregate([.9,.8,.7],[np.eye(6)[1],np.eye(6)[2],np.eye(6)[3]],'six')['prediction'],1)
        with self.assertRaises(ValueError):aggregate([],[],'binary')

    def test_fixed_models_and_training_step(self):
        for task in CLASSES:
            set_random_seed(3407,True)
            model=build(task,checkpoint=fixed_checkpoint(task))
            model.eval();x=torch.randn(2,3,224,224)
            with torch.no_grad():y=model(x,return_loss=False)
            self.assertEqual(tuple(y.shape),(2,len(CLASSES[task])))
            torch.testing.assert_close(y.sum(1),torch.ones(2))
            teacher={k:v.clone() for k,v in model.state_dict().items() if k.startswith('cam_generator.')}
            model.train();opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-5)
            opt.zero_grad()
            _,loss=model(x,targets=torch.arange(2),return_loss=True,train_statu=True)
            self.assertTrue(torch.isfinite(loss['loss']))
            loss['loss'].backward()
            self.assertIsNotNone(model.head.fc.weight.grad)
            self.assertTrue(torch.isfinite(model.head.fc.weight.grad).all())
            opt.step()
            if task!='view':
                self.assertFalse(model.cam_generator.training)
                for k,v in teacher.items():self.assertTrue(torch.equal(model.state_dict()[k],v))
                self.assertTrue(all(p.grad is None for p in model.cam_generator.parameters()))
                self.assertIsNotNone(model.paper_branch.zero_conv.weight.grad)

    def test_manifest_and_pipelines(self):
        with tempfile.TemporaryDirectory(prefix='echocat_fixtures_') as tmp:
            d=Path(tmp);image=d/'synthetic image.png'
            Image.fromarray(np.random.default_rng(23).integers(0,256,(270,310,3),dtype=np.uint8)).save(image)
            manifest=d/'list.txt';manifest.write_text('synthetic image.png 1\n')
            lines=load_lines(manifest,2)
            for task in CLASSES:
                c=recipe(task)
                for split in ['train_pipeline','val_pipeline']:
                    set_random_seed(3407,True)
                    sample=Mydataset(lines,c[split])[0][0]
                    expected=(2,3,224,224) if task=='view' and split=='train_pipeline' else (3,224,224)
                    self.assertEqual(tuple(sample.shape),expected)
                    self.assertTrue(torch.isfinite(sample).all())

    def test_cli_fetal_cam_teacher_and_workers(self):
        with tempfile.TemporaryDirectory(prefix='echocat_interfaces_') as tmp:
            d=Path(tmp);env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            for i in range(4):
                Image.fromarray(np.random.default_rng(i).integers(0,256,(270,290,3),dtype=np.uint8)).save(d/f'{i}.png')
            fetal=d/'fetal.jsonl'
            fetal.write_text('\n'.join(json.dumps({'image':f'{i}.png','fetus_id':f'case{i//2}',
                                                 'fetus_label':i//2}) for i in range(4)))
            for task in ['binary','six']:
                args=[sys.executable,str(ROOT/'fetal.py'),'--task',task,'--manifest',str(fetal),
                      '--output',str(d/f'{task}_fetal.json'),'--device','cpu','--batch-size','2']
                r=subprocess.run(args,cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
                self.assertEqual(r.returncode,0,r.stderr[-4000:])
                self.assertEqual(len(json.loads((d/f'{task}_fetal.json').read_text())['records']),2)
                r=subprocess.run([sys.executable,str(ROOT/'gradcam.py'),'--task',task,
                    '--image',str(d/'0.png'),'--output',str(d/f'cam_{task}'),'--device','cpu'],
                    cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
                self.assertEqual(r.returncode,0,r.stderr[-4000:])
                with Image.open(d/f'cam_{task}/gradcam.png') as rendered:
                    self.assertEqual(rendered.size,(224,224))
            train=d/'train.txt';val=d/'val.txt'
            train.write_text('0.png 0\n1.png 1\n');val.write_text('2.png 0\n3.png 1\n')
            from torch.utils.data import DataLoader
            from utils.dataloader import collate
            loader=DataLoader(Mydataset(load_lines(train,2),recipe('view')['train_pipeline']),
                              batch_size=2,num_workers=4,collate_fn=collate)
            self.assertEqual(tuple(next(iter(loader))[0].shape),(2,2,3,224,224))
            r=subprocess.run([sys.executable,str(ROOT/'train_teacher.py'),'--train-list',str(train),
                '--val-list',str(val),'--output',str(d/'teacher'),'--epochs','1','--batch-size','2',
                '--workers','0','--device','cpu','--deterministic'],cwd=ROOT,env=env,
                capture_output=True,text=True,timeout=90)
            self.assertEqual(r.returncode,0,r.stderr[-4000:])
            model=build('binary',initialize=True,teacher=d/'teacher/best_resnet18_view_gray.pth')
            self.assertFalse(model.cam_generator.training)

    def test_real_cli_best_last_and_resume(self):


        with tempfile.TemporaryDirectory(prefix='echocat_train_') as tmp:
            d=Path(tmp)
            for i in range(12):
                a=np.random.default_rng(i).integers(0,256,(256,256,3),dtype=np.uint8)
                Image.fromarray(a).save(d/f'{i}.png')
            env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            for task in CLASSES:
                k=len(CLASSES[task]);train=d/f'{task}_train.txt';val=d/f'{task}_val.txt'
                train.write_text(''.join(f'{i}.png {i%k}\n' for i in range(6)))
                val.write_text(''.join(f'{i}.png {i%k}\n' for i in range(6,12)))
                output=d/task
                cmd=[sys.executable,str(ROOT/'train.py'),'--task',task,'--train-list',str(train),
                     '--val-list',str(val),'--output',str(output),'--epochs','2','--workers','0',
                     '--batch-size','2','--device','cpu']
                r=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=360)
                self.assertEqual(r.returncode,0,r.stderr[-6000:]+r.stdout[-2000:])
                index=json.loads((output/'checkpoints.json').read_text())
                self.assertEqual({x.name for x in output.glob('*.pth')},{index['best_file'],index['last_file']})
                history=[json.loads(x) for x in (output/'history.jsonl').read_text().splitlines()]
                self.assertEqual(len(history),2)
                checkpoint=torch.load(output/index['best_file'],weights_only=True,map_location='cpu')
                self.assertEqual(tuple(checkpoint['selection_key']),max(tuple(x['selection_key']) for x in history))
                model=build(task,checkpoint=output/index['best_file']);self.assertEqual(model.head.num_classes,k)
                last=torch.load(output/index['last_file'],weights_only=True,map_location='cpu')
                self.assertEqual(last['epoch'],2)
                self.assertEqual(json.loads(json.dumps(last['validation'])),history[-1]['validation'])
                if index['last_epoch']!=index['best_epoch']:
                    self.assertEqual(last['checkpoint_kind'],'last')
                last_model=build(task,checkpoint=output/index['last_file'])
                self.assertEqual(last_model.head.num_classes,k)
                r=subprocess.run(cmd+['--resume-best',str(output/index['best_file'])],cwd=ROOT,env=env,
                                 capture_output=True,text=True,timeout=360)
                self.assertEqual(r.returncode,0,r.stderr[-4000:])
                infer=subprocess.run([sys.executable,str(ROOT/'predict.py'),'--task',task,
                    '--image',str(d/'0.png'),'--checkpoint',str(output/index['best_file']),
                    '--output',str(d/f'{task}_pred.json'),'--device','cpu'],
                    cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
                self.assertEqual(infer.returncode,0,infer.stderr[-4000:])


if __name__=='__main__':unittest.main(verbosity=2)
