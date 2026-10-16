import unittest

import torch
from torch import nn

import contract as c
from train_phase5 import configure, agreement_weights, assert_frozen


class LossAblationTests(unittest.TestCase):
    def test_gold_and_teacher_gating_uses_only_training_reference(self):
        y=torch.tensor([0,1])
        gold=torch.tensor([[.9,.1],[.9,.1]])
        teacher=torch.tensor([[.1,.9],[.1,.9]])
        g,t=agreement_weights(y,gold,teacher,2,.15,True,True)
        self.assertEqual(g.tolist(),[2,0])
        self.assertTrue(torch.allclose(t,torch.tensor([0,.15])))

    def test_ungated_control_does_not_change_labels(self):
        y=torch.tensor([0,1]);p=torch.tensor([[.9,.1],[.9,.1]])
        before=y.clone();g,t=agreement_weights(y,p,p,.1,.15,False,False)
        self.assertTrue(torch.equal(y,before))
        self.assertTrue(torch.allclose(g,torch.tensor([.1,.1])))
        self.assertTrue(torch.allclose(t,torch.tensor([.15,.15])))

    def test_last_two_blocks_preserve_original_network_and_buffers(self):
        model=c.load_model(c.ROOT/'start.pth')
        state={k:v.clone() for k,v in model.state_dict().items()}
        params=configure(model,2)
        self.assertEqual(c.parameter_count(model),1143751)
        self.assertEqual(sum(p.numel() for p in params),759445)
        self.assertTrue(all(n.startswith(('res_layers.3.','res_layers.4.','head_rhythm.')) for n,p in model.named_parameters() if p.requires_grad))
        self.assertTrue(all(not m.training for m in model.modules() if isinstance(m,nn.BatchNorm1d)))
        self.assertEqual(assert_frozen(model,state,2),[])
        with torch.no_grad():model.stem[0].weight.add_(1)
        with self.assertRaisesRegex(ValueError,'Frozen'):
            assert_frozen(model,state,2)


if __name__=='__main__':unittest.main()
