# ECGFounder Source Notice

`net1d.py` is copied without modification from the
[official ECGFounder repository](https://github.com/PKUDigitalHealth/ECGFounder)
at commit `04edac702b61c91face519774ddcc0cd712fef23`.

Source SHA-256: `832bf90e05e20e62db6f9c7007acf7f0b1f739938ab80fa9b2c1ad438e8da7f7`.
The accompanying upstream MIT license is retained as `LICENSE`.
The source includes its original Shenda Hong attribution.

The public pretrained weights are separately downloaded from
[the official model repository](https://huggingface.co/PKUDigitalHealth/ECGFounder)
at revision `d9b1793951b2342f5f7e84f1ac03cd37f8a08724`; its model card declares MIT.
Their file SHA-256 is `ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997`.
Those weights are not part of this source copy or redistributed in Git.

The upstream training/data loaders and evaluation threshold selection are not
adopted here. TITAN uses its own paired contracts, validation-only selection
and observed-target masks. Stored upstream AUROC is not a TITAN result.
