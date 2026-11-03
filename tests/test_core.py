import json
import unittest
import tempfile
from pathlib import Path
import numpy as np
from weight_space_breakout.game import Breakout
from weight_space_breakout.policy import Policy, FAULTS, apply_edits, faulty_policy, tracking_policy, repair_edits, parse_submission
from weight_space_breakout.evaluation import rollout, score_submission, PRACTICE_SEEDS, TEST_SEEDS


class PhysicsTests(unittest.TestCase):
    def test_shape_border_and_memory(self):
        e = Breakout()
        x, _ = e.reset(42)
        grid = x.reshape(10,10,5)
        self.assertEqual(x.shape, (500,))
        self.assertEqual(grid[:,:,3].sum(), 36)
        self.assertEqual(grid[:,:,0].sum(), 3)
        self.assertEqual(grid[:,:,2].sum(), 24)
        self.assertEqual(grid[:,:,4].sum(), 0)
        y, *_ = e.step(4)
        np.testing.assert_array_equal(y.reshape(10,10,5)[:,:,4], np.maximum(grid[:,:,0], grid[:,:,1]))

    def test_wall_bounce(self):
        e = Breakout(); e.row=5; e.col=1; e.dx=-1; e.dy=1
        e.step(0)
        self.assertEqual((e.row,e.col,e.dx),(6,2,1))

    def test_brick_reward_once(self):
        e = Breakout(); e.row=4; e.col=4; e.dy=-1; e.dx=1
        _,reward,*_=e.step(0)
        self.assertEqual(reward,1)
        self.assertNotIn((3,4),e.bricks)
        self.assertEqual(e.dy,1)

    def test_paddle_hit_and_miss(self):
        e = Breakout(); e.row=7; e.col=4; e.dy=1; e.dx=1; e.paddle=5
        _,r,done,_,_=e.step(0)
        self.assertFalse(done); self.assertEqual(e.dy,-1); self.assertEqual(r,0)
        e.reset(); e.row=7; e.col=4; e.dy=1; e.dx=1; e.paddle=2
        _,r,done,_,info=e.step(0)
        self.assertTrue(done); self.assertEqual(r,-1); self.assertEqual(info['reason'],'miss')
        with self.assertRaises(RuntimeError): e.step(0)

    def test_win_and_time_limit(self):
        e=Breakout(); e.bricks={(3,4)}; e.row=4; e.col=4; e.dy=-1; e.dx=1
        _,r,done,trunc,info=e.step(0)
        self.assertTrue(done); self.assertFalse(trunc); self.assertEqual(info['reason'],'cleared'); self.assertEqual(r,1)
        e=Breakout(max_steps=1)
        _,_,done,trunc,info=e.step(0)
        self.assertTrue(trunc); self.assertFalse(done); self.assertEqual(info['reason'],'time_limit')

    def test_invariants_and_reproducibility(self):
        p=tracking_policy()
        a=rollout(p,17,record=True); b=rollout(p,17,record=True)
        self.assertEqual(a,b)
        for seed in range(200):
            e=Breakout(); x,_=e.reset(seed); rng=np.random.default_rng(seed)
            while not(e.terminated or e.truncated):
                x,r,*_=e.step(p.act(x,rng))
                self.assertTrue(1<=e.row<=8 and 1<=e.col<=8)
                self.assertNotIn((e.row,e.col),e.bricks)
                self.assertEqual(e.total,24-len(e.bricks)-(e.reason=='miss'))
                self.assertTrue(np.isin(x,[0,1]).all())


class RepairTests(unittest.TestCase):
    def test_no_edits_and_gold(self):
        self.assertFalse(set(PRACTICE_SEEDS)&set(TEST_SEEDS))
        for fault in FAULTS:
            self.assertEqual(score_submission('{"edits":[]}',fault)['reward'],0)
            score=score_submission(json.dumps(repair_edits(fault)),fault)
            self.assertEqual(score['reward'],1)
            self.assertGreater(score['gain'],0)

    def test_invalid_submissions(self):
        bad=['not json','{"edits":NaN}','{"edits":[],"edits":[]}',
             '{"edits":[{"tensor":"W1","index":[-1,0],"value":1}]}',
             '{"edits":[{"tensor":"W1","index":[0,0],"value":1e999}]}',
             '{"edits":[{"tensor":"b1","index":[true],"value":1}]}']
        for text in bad:
            self.assertFalse(score_submission(text,FAULTS[0])['valid'])

    def test_export_and_independent_forward(self):
        p=tracking_policy(); x,_=Breakout().reset(7)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'policy.npz'; p.save(path); q=Policy.load(path)
            np.testing.assert_array_equal(p.logits(x),q.logits(x))
        w=p.weights
        a=np.tanh(np.einsum('i,ij->j',x,w['W1'])+w['b1'])
        b=np.tanh(np.einsum('i,ij->j',a,w['W2'])+w['b2'])
        np.testing.assert_allclose(p.logits(x),b@w['W3']+w['b3'],atol=1e-5)
        self.assertAlmostEqual(float(p.probabilities(x).sum()),1,places=6)


if __name__=='__main__': unittest.main()
