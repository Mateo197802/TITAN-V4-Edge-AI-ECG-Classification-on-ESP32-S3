# References

References added for the dataset families explicitly named by the validation table and for the preprocessing/deployment interfaces documented by this repository. They describe source datasets and software; they are not evidence that the specific records in this checkout came from those exact versions.

## ECG Datasets

1. Wagner P, Strodthoff N, Bousseljot R-D, Kreiseler D, Lunze F I, Samek W, Schaeffter T. PTB-XL, a large publicly available ECG dataset. *Scientific Data*. 2020;7:154. [https://doi.org/10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6). PhysioNet currently lists release 1.0.3 with version DOI [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45) and latest-version DOI [10.13026/6sec-a640](https://doi.org/10.13026/6sec-a640); the release page states CC BY 4.0 for its files. The version used to create this repository's checkpoint is not recorded.
2. Reyna MA, Sadr N, Perez Alday EA, Gu A, Shah AJ, Robichaux C, Rad AB, Elola A, Seyedi S, Ansari S, Ghanbari H, Li Q, Sharma A, Clifford GD. Will Two Do? Varying Dimensions in Electrocardiography: The PhysioNet/Computing in Cardiology Challenge 2021. *Computing in Cardiology*. 2021;48:1-4. [https://doi.org/10.23919/CinC53138.2021.9662687](https://doi.org/10.23919/CinC53138.2021.9662687). The versioned PhysioNet release is 1.0.3 (2022), DOI [10.13026/34va-7q14](https://doi.org/10.13026/34va-7q14), with CC BY 4.0 terms on the release page. Its training-source list includes CPSC/CPSC-Extra, Georgia, Chapman-Shaoxing, Ningbo, PTB, and PTB-XL.
3. Zheng J, Zhang J, Danioko S, Yao H, Guo H, Rakovski C. A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients. *Scientific Data*. 2020;7:48. [https://doi.org/10.1038/s41597-020-0386-x](https://doi.org/10.1038/s41597-020-0386-x). This describes the Chapman-Shaoxing dataset family.
4. Zheng J, Chu H, Struppa D, Zhang J, Yacoub SM, El-Askary H, et al. Optimal multi-stage arrhythmia classification approach. *Scientific Reports*. 2020;10:2898. [https://doi.org/10.1038/s41598-020-59821-7](https://doi.org/10.1038/s41598-020-59821-7). This is cited by the PhysioNet Challenge source documentation for Ningbo.
5. Perez Alday EA, Gu A, Shah AJ, Robichaux C, Wong AKI, Liu C, et al. Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020. *Physiological Measurement*. 2021;41(12):124003. [https://doi.org/10.1088/1361-6579/abc960](https://doi.org/10.1088/1361-6579/abc960). The challenge release documents the CPSC 2018/CPSC-Extra and Georgia sources.
6. China Physiological Signal Challenge 2018. Official challenge site: [http://2018.icbeb.org/Challenge.html](http://2018.icbeb.org/Challenge.html). The repository does not document the precise CPSC 2018 files or standalone license used to produce its rows; consult the original source terms before redistributing data.

Official dataset landing pages: [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/), [PhysioNet Challenge 2021 1.0.3](https://physionet.org/content/challenge-2021/1.0.3/), and [PhysioNet Challenge 2020 1.0.2](https://physionet.org/content/challenge-2020/1.0.2/). PhysioNet's current recommended repository citation is Pollard T, Moody BE, Lehman L, et al. *PhysioNet as a global platform for biomedical research*. *Nature Health*. 2026;1(8):792-795. [https://doi.org/10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z). The unresolved `data_test` group has no source citation in this checkout.

## Software and Signal Processing

- SciPy `resample_poly` API: [https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html).
- Paramiko SSH client and host-key policy API: [https://docs.paramiko.org/en/latest/api/client.html](https://docs.paramiko.org/en/latest/api/client.html).
- PlatformIO Espressif32 platform configuration and version pinning: [https://docs.platformio.org/en/latest/platforms/espressif32.html](https://docs.platformio.org/en/latest/platforms/espressif32.html).

## Citation Scope

These citations document dataset provenance and referenced software behavior. They do not constitute the missing manuscript citation for TITAN V4, validate model claims, or establish which data release was used for training. The repository has no associated manuscript DOI/reference at this time.
