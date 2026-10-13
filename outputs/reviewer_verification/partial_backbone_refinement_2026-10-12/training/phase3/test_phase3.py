import unittest

import torch
from torch import nn

import contract as c
from train_phase3 import ALLOWED_PREFIXES, configure_partial


class PartialBackboneContractTests(unittest.TestCase):
    def test_warm_start_has_original_parameter_count(self):
        model = c.load_model(c.ROOT / "phase2_selected.pth")
        self.assertEqual(c.parameter_count(model), c.PARAMETERS)
        self.assertEqual(c.PARAMETERS, 1_143_751)

    def test_only_original_final_block_and_rhythm_head_are_trainable(self):
        model = c.load_model(c.ROOT / "phase2_selected.pth")
        trainable = configure_partial(model)
        names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertTrue(names)
        self.assertTrue(all(name.startswith(ALLOWED_PREFIXES) for name in names))
        self.assertEqual(sum(parameter.numel() for parameter in trainable), 460_065)

    def test_batch_norm_statistics_are_frozen_and_dropout_is_train_only(self):
        model = c.load_model(c.ROOT / "phase2_selected.pth")
        configure_partial(model)
        batch_norms = [module for module in model.res_layers[4].modules() if isinstance(module, nn.BatchNorm1d)]
        dropouts = [module for module in model.res_layers[4].modules() if isinstance(module, nn.Dropout)]
        self.assertTrue(batch_norms)
        self.assertTrue(all(not module.training for module in batch_norms))
        self.assertTrue(dropouts)
        self.assertTrue(all(module.training for module in dropouts))

    def test_original_inference_contract_is_unchanged(self):
        model = c.load_model(c.ROOT / "phase2_selected.pth").eval()
        with torch.inference_mode():
            outputs = model(torch.zeros(2, 6, 1250))
        self.assertEqual(len(outputs), 4)
        self.assertEqual(tuple(outputs[0].shape), (2, 9))
        self.assertEqual(tuple(outputs[1].shape), (2, 1))
        self.assertEqual(tuple(outputs[2].shape), (2, 3))
        self.assertEqual(tuple(outputs[3].shape), (2, 10))


if __name__ == "__main__":
    unittest.main()
