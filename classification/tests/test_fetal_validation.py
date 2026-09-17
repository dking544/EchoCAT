import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from echocat.fetal_validation import load_case_map, pooled_metrics, validate_fetal, path_key
from echocat.metrics import classification_metrics


class FetalValidationTests(unittest.TestCase):
    def test_mapping_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mapping=root/'cases.jsonl'
            lines=[f'{root / "a.png"} 0\n', f'{root / "b.png"} 1\n']
            good=[{'image':'a.png','fetus_id':'a'}, {'image':'b.png','fetus_id':'b'}]
            def write(rows):mapping.write_text('\n'.join(json.dumps(x) for x in rows))
            write(good);self.assertEqual(len(load_case_map(mapping,lines)),2)
            bad=[good[:1], good+good[:1], good+[{'image':'c.png','fetus_id':'c'}],
                 [{'image':'a.png','fetus_id':'a'},{'image':'b.png','fetus_id':'a'}],
                 [{'image':'a.png','fetus_id':''},good[1]],
                 [{'image':'a.png','fetus_id':'a','fetus_label':1},good[1]]]
            for rows in bad:
                write(rows)
                with self.assertRaises(ValueError):load_case_map(mapping,lines)

    def test_pool_matches_historical_formula(self):
        paths=[str(Path(f'example_{i}.png').resolve()) for i in range(12)]
        labels=torch.arange(6).repeat_interleave(2)
        scores=torch.eye(6)[labels]*.8+.2/6
        scores[1]=torch.tensor([.01,.94,.01,.01,.01,.02])
        case_map={path_key(p):f'case{i//2}' for i,p in enumerate(paths)}
        expected=torch.stack([torch.softmax(scores,dim=1)[2*i:2*i+2].mean(0) for i in range(6)])
        ref=classification_metrics(expected.numpy(),np.arange(6))
        actual=pooled_metrics(scores,labels,paths,case_map)
        self.assertEqual(actual['total_samples'],6)
        self.assertEqual(actual['confusion_matrix'],ref['confusion_matrix'])
        self.assertEqual(actual['f1_macro'],ref['f1_macro'])
        with self.assertRaises(ValueError):pooled_metrics(scores,labels,paths[:-1],case_map)
        with self.assertRaises(ValueError):pooled_metrics(scores[:-2],labels[:-2],paths[:-2],dict(list(case_map.items())[:-2]))

    def test_validation_forward_and_grouping(self):
        labels=torch.arange(6)
        paths=[str(Path(f'synthetic_{i}.png').resolve()) for i in range(6)]
        mapping={path_key(p):f'case{i}' for i,p in enumerate(paths)}
        class Model(torch.nn.Module):
            def forward(self,x,return_loss=False):
                if return_loss:raise AssertionError('Wrong validation forward')
                return x
        result=validate_fetal(Model(),[(torch.eye(6),labels,paths)],'cpu',mapping)
        self.assertEqual(result['fetal']['f1_macro'],1.)
        self.assertEqual(result['fetal']['total_samples'],6)
        self.assertGreater(result['avg_loss'],0.)


if __name__ == '__main__':unittest.main(verbosity=2)
