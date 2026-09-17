import unittest
from echocat.monitoring import EarlyStopMonitor


class MonitorTests(unittest.TestCase):
    def test_three_epoch_mean_and_restore_epoch(self):
        m = EarlyStopMonitor('macro_precision_3mean', 'binary')
        records = [m.update(v, i+1) for i, v in enumerate([.1,.2,.3] + [0.]*10)]
        self.assertIsNone(records[0]['score'])
        self.assertIsNone(records[1]['score'])
        self.assertEqual(m.best_epoch, 3)
        self.assertFalse(records[-2]['should_stop'])
        self.assertTrue(records[-1]['should_stop'])
        self.assertEqual(m.epoch, 13)

    def test_fetal_ties_and_patience(self):
        m = EarlyStopMonitor('fetal_macro_f1', 'six')
        for e in range(1, 16):
            self.assertFalse(m.update(.5, e)['should_stop'])
        self.assertTrue(m.update(.5, 16)['should_stop'])
        self.assertEqual(m.best_epoch, 1)

    def test_resume_best_window(self):
        m = EarlyStopMonitor('macro_precision_3mean', 'binary')
        for e,v in enumerate([.1,.2,.3],1):m.update(v,e)
        resumed = EarlyStopMonitor(m.name, 'binary')
        resumed.load_state_dict(m.state_dict())
        for e,v in enumerate([.2,.1,0.,0.,0.],4):
            self.assertEqual(m.update(v,e),resumed.update(v,e))

    def test_invalid_values_and_task(self):
        with self.assertRaises(ValueError):EarlyStopMonitor('fetal_macro_f1','binary')
        m = EarlyStopMonitor('fetal_macro_f1','six')
        for v in [float('nan'),float('inf'),-.1,1.1]:
            with self.assertRaises(ValueError):m.update(v,1)
        with self.assertRaises(ValueError):m.update(.5,2)


if __name__ == '__main__':unittest.main(verbosity=2)
