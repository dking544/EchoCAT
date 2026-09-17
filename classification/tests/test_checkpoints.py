import json
from pathlib import Path
import tempfile
import unittest
import torch
from echocat.checkpoints import EpochCheckpoints


class CheckpointTests(unittest.TestCase):
    def payload(self, epoch):
        return {'epoch':epoch,'state_dict':{'weight':torch.tensor([epoch])}}

    def test_retain_best_and_final_and_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=EpochCheckpoints(root)
            store.save(self.payload(1),best=True)
            store.save(self.payload(3),best=True)
            self.assertFalse((root/'epoch_1.pth').exists())
            store.save(self.payload(8),last=True)
            self.assertEqual({p.name for p in root.glob('*.pth')},{'epoch_3.pth','epoch_8.pth'})
            resumed=EpochCheckpoints(root,root/'epoch_3.pth')
            resumed.save(self.payload(9),best=True,last=True)
            self.assertEqual([p.name for p in root.glob('*.pth')],['epoch_9.pth'])
            self.assertEqual(json.loads((root/'checkpoints.json').read_text())['last_file'],'epoch_9.pth')

    def test_coincident_epochs_save_one_reloadable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=EpochCheckpoints(root)
            store.save(self.payload(3),best=True,last=True)
            self.assertEqual([p.name for p in root.glob('*.pth')],['epoch_3.pth'])
            payload=torch.load(store.best_path,weights_only=True)
            self.assertEqual(payload['state_dict']['weight'].item(),3)
            self.assertEqual(EpochCheckpoints(root,store.best_path).last_epoch,3)

    def test_reject_existing_untracked_and_wrong_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=EpochCheckpoints(root)
            store.save(self.payload(2),best=True)
            store.save(self.payload(4),last=True)
            with self.assertRaises(FileExistsError):EpochCheckpoints(root)
            with self.assertRaises(ValueError):EpochCheckpoints(root,root/'epoch_4.pth')
            (root/'unrelated.pth').write_bytes(b'unrelated')
            with self.assertRaises(ValueError):EpochCheckpoints(root,root/'epoch_2.pth')
            self.assertEqual((root/'unrelated.pth').read_bytes(),b'unrelated')

    def test_index_cannot_reference_other_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=EpochCheckpoints(root)
            store.save(self.payload(2),best=True)
            record=json.loads((root/'checkpoints.json').read_text())
            record['best_file']='../unrelated.pth'
            (root/'checkpoints.json').write_text(json.dumps(record))
            with self.assertRaises(ValueError):EpochCheckpoints(root,root/'epoch_2.pth')
            with self.assertRaises(ValueError):store.save(self.payload(-1),best=True)


if __name__=='__main__':unittest.main(verbosity=2)
