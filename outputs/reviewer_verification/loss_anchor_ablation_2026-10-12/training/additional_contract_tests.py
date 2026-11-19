import unittest
import ast
from pathlib import Path

import torch
from torch import nn

import contract as c
from train_phase5 import configure, agreement_weights, assert_frozen, PROTOCOL, validate_data


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

    def test_control_pair_differs_only_in_gold_anchor(self):
        first,second=PROTOCOL['optimization']['trials'][:2]
        self.assertEqual({k:v for k,v in first.items() if k not in ('name','gold_weight')},
                         {k:v for k,v in second.items() if k not in ('name','gold_weight')})
        self.assertEqual(first['gold_weight'],2)
        self.assertEqual(second['gold_weight'],0)

    def test_followup_labels_loaded_after_selection_is_saved(self):
        tree=ast.parse((Path(__file__).parent/'train_phase5.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        events=[]
        for node in ast.walk(main):
            if isinstance(node,ast.Call):
                constants=[n.value for n in ast.walk(node) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
                if isinstance(node.func,ast.Attribute) and node.func.attr=='save' and 'selection.json' in constants:
                    events.append(('selection',node.lineno))
                if isinstance(node.func,ast.Attribute) and node.func.attr=='load' and 'evaluation.npz' in constants:
                    events.append(('evaluation',node.lineno))
        self.assertEqual(len(events),2)
        self.assertLess(dict(events)['selection'],dict(events)['evaluation'])

    def test_original_four_head_inference_shapes(self):
        model=c.load_model(c.ROOT/'start.pth')
        torch.set_num_threads(2)
        with torch.inference_mode():values=model(torch.zeros(2,6,1250))
        self.assertEqual([tuple(v.shape) for v in values],[(2,9),(2,1),(2,3),(2,10)])


if __name__=='__main__':unittest.main()
