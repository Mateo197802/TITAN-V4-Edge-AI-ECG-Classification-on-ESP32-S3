# References and Dataset Attribution

The frozen evaluation records in this repository were resolved to the `training/` files of the PhysioNet/Computing in Cardiology Challenge 2021 release 1.0.3. Cite the exact release and its primary Challenge paper. The original dataset papers below identify the component source families; they are not evidence that the TITAN checkpoint was trained on any particular record or release.

## Data Releases and Primary Dataset Papers

1. Reyna MA, Sadr N, Perez Alday EA, Gu A, Liu C, Seyedi S, Shah A, Clifford GD. Will Two Do? Varying Dimensions in Electrocardiography: The PhysioNet/Computing in Cardiology Challenge 2021. *Computing in Cardiology*. 2021;48:1-4. [doi:10.23919/CinC53138.2021.9662687](https://doi.org/10.23919/CinC53138.2021.9662687). Exact data release used for evaluation: PhysioNet version 1.0.3 (2022), [doi:10.13026/34va-7q14](https://doi.org/10.13026/34va-7q14), [release page](https://physionet.org/content/challenge-2021/1.0.3/). The release page identifies CC BY 4.0 for files and lists CPSC/CPSC-Extra, Georgia, PTB/PTB-XL, Chapman-Shaoxing, Ningbo, and other source families.
2. Wagner P, Strodthoff N, Bousseljot R-D, Kreiseler D, Lunze F I, Samek W, Schaeffter T. PTB-XL, a large publicly available ECG dataset. *Scientific Data*. 2020;7:154. [doi:10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6). The Challenge release cites an earlier PTB-XL parent release; its contents are accessed here through the Challenge 2021 v1.0.3 package, not by substituting the separately versioned PTB-XL 1.0.3 release. For reference, the latter has version DOI [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45) and its own [release page](https://physionet.org/content/ptb-xl/1.0.3/).
3. Zheng J, Zhang J, Danioko S, Yao H, Guo H, Rakovski C. A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients. *Scientific Data*. 2020;7:48. [doi:10.1038/s41597-020-0386-x](https://doi.org/10.1038/s41597-020-0386-x). Chapman-Shaoxing dataset.
4. Zheng J, Chu H, Struppa D, Zhang J, Yacoub SM, El-Askary H, et al. Optimal multi-stage arrhythmia classification approach. *Scientific Reports*. 2020;10:2898. [doi:10.1038/s41598-020-59821-7](https://doi.org/10.1038/s41598-020-59821-7). Ningbo dataset source paper cited by the Challenge documentation.
5. Perez Alday EA, Gu A, Shah AJ, Robichaux C, Wong AKI, Liu C, et al. Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020. *Physiological Measurement*. 2021;41(12):124003. [doi:10.1088/1361-6579/abc960](https://doi.org/10.1088/1361-6579/abc960). Challenge documentation for CPSC 2018/CPSC-Extra and Georgia sources.
6. China Physiological Signal Challenge 2018. [Official challenge site](http://2018.icbeb.org/Challenge.html). The TITAN evaluation uses the versioned Challenge 2021 training bundle described above; do not substitute a standalone CPSC download when reproducing it.
7. Pollard T, Moody BE, Lehman L, Gow B, Fernandes C, Xie C, Johnson A, Mark RG, Heldt T. PhysioNet as a global platform for biomedical research. *Nature Health*. 2026;1(8):792-795. [doi:10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z). Current recommended general PhysioNet citation.

These publication dates are not obsolete citations: the original papers identify the datasets and methods, while the versioned PhysioNet DOI identifies the exact release downloaded by this workflow. For the manuscript, cite both the original Challenge paper and its exact 1.0.3 release; cite relevant component-dataset papers as appropriate.

## Methods and Tooling

- PhysioNet/CinC Challenge 2021 official evaluation code and scoring files: [evaluation-2021](https://github.com/physionetchallenges/evaluation-2021). Challenge recordings may have one or more labels; its official weighted score is not ordinary single-label accuracy.
- SciPy `resample_poly`: [API documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html).
- WFDB Python package: [documentation](https://wfdb.readthedocs.io/).
- Paramiko SSH client and host-key policy: [API documentation](https://docs.paramiko.org/en/latest/api/client.html).
- PlatformIO Espressif32 platform: [platform documentation](https://docs.platformio.org/en/latest/platforms/espressif32.html).
- Legacy Arduino TFLite Micro dependency used by the checked-in firmware: [TensorFlowLite_ESP32, pinned at commit `e88e0eb`](https://github.com/tanakamasayuki/Arduino_TensorFlowLite_ESP32/tree/e88e0ebee0430ed716ff5b49854795db90066e59), Apache-2.0. Its maintainer notes that it is outdated and no longer recommended; the [Espressif maintained component](https://github.com/espressif/esp-tflite-micro) targets ESP-IDF and is not yet integrated by this Arduino firmware.

## Citation Scope

Dataset references establish upstream attribution and release metadata. They do not establish TITAN training provenance, train/test independence, clinical validity, or an associated TITAN manuscript DOI. See [DATA_PROVENANCE.md](DATA_PROVENANCE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact claim boundary.
