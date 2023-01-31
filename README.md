# Interaction Mix and Match: Synthesizing Close Interaction using Conditional Hierarchical GAN with Multi-Hot Class Embedding ( SCA 2022, to be published in Computer Graphics Forum )

Aman Goel, Qianhui Men, [Edmond S. L. Ho](http://www.edho.net/)

Synthesizing multi-character interactions is a challenging task due to the complex and varied interactions between the characters. In particular, precise spatiotemporal alignment between characters is required in generating close interactions such as dancing and fighting. Existing work in generating multi-character interactions focuses on generating a single type of reactive motion for a given sequence which result in a lack of variety of the resultant motions. In this paper, we propose a novel way to create realistic skeleton human reactive motions which are not presented in the given dataset by mixing and matching reactive motions. We propose a Conditional Hierarchical Generative Adversarial Network with Multi-Hot Class Embedding to generate the Mix and Match reactive motions from a given input active motion of a sequence. Experiments are conducted on both noisy (depth-based) and high-quality (MoCap-based) interaction datasets. The quantitative and qualitative comparison results show that our approach outperforms the state-of-the-art methods on the given datasets. We also provide an augmented dataset with realistic reactive motions with flexible patterns to stimulate future research in this area.

## [Paper](https://arxiv.org/abs/2208.00774) | [Demo](https://www.youtube.com/watch?v=RhzNEFM7wbY)
<br/>
                                                                                                                                                                                                                                                                                                                          
![framework_new](https://user-images.githubusercontent.com/109843145/182305175-1ea634a8-911e-402c-b4c0-18395503a0ab.jpg)

## Environment

This project is developed and tested on Ubuntu 20.04, Python 3.0+, and Tensorflow 1.5.0. Since the repository is developed based on [MSHL22](https://www.sciencedirect.com/science/article/abs/pii/S0097849321002089?via%3Dihub) of Men et al., the environment requirements, installation and dataset preparation process generally follow theirs.

## Installation

1. Clone this repository

      ```git clone https://github.com/Aman-Goel1/IMM```

2. Download the [preprocessed SBU dataset](https://drive.google.com/drive/folders/1bRlXjdQJF0MLIa6FvRSc3l0R3IAjkMRi?usp=sharing) and extract it to

      ```./datasets/SBU/normalized_7_fold/```

## Training

``` python3 trainsbu.py```

Note on reproducibility:
Since we didn't fix a random seed, you might not be able to reproduce the same AFD in the paper. But, several runs with different random seeds fell in a similar AFD range.


## Matlab animation


Place the motion sequences in ```/motion``` and run ```drawskt_SBU.m```

## Synthetic Dataset

Download the [synthetic dataset](https://drive.google.com/drive/folders/1wWW7uIviILIzMFldrojjPk1ahu2o6C7G?usp=sharing). The dataloader has been updated to also load in the synthetic dataset


## License

Please see ```License```

## Citation

If you find our work useful in your research, please consider citing
```
@article {10.1111:cgf.14647,
journal = {Computer Graphics Forum},
title = {{Interaction Mix and Match: Synthesizing Close Interaction using Conditional Hierarchical GAN with Multi-Hot Class Embedding}},
author = {Goel, Aman and Men, Qianhui and Ho, Edmond S. L.},
year = {2022},
publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
ISSN = {1467-8659},
DOI = {10.1111/cgf.14647}
}
```

Please feel free to contact us (aman.goel@students.iiit.ac.in) with any questions or concerns.

## Acknowledgement

The codebase is developed based on [MSHL22](https://www.sciencedirect.com/science/article/abs/pii/S0097849321002089?via%3Dihub) of Men et al.
