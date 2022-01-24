# gesture-generation-using-WGAN

## Introduction

Official implementation for "Probabilistic Human-like Gesture Synthesis from Speech using GRU-based WGAN"
https://dl.acm.org/doi/10.1145/3461615.3485407
https://openreview.net/forum?id=ykvm7OLh7B.

## Example

![A simple sample for wgan on TAKEKUCHI dataset](demo/cl34-dev8.gif)

## Usage

### To reproduce
1. Clone the repository
2. execute 'conda env create -f environment.yml' to create a conda environment
3. Download a dataset from https://www.dropbox.com/sh/j419kp4m8hkt9nd/AAC_pIcS1b_WFBqUp5ofBG1Ia?dl=0
4. Download the extracted speech features https://drive.google.com/drive/folders/1eISYiVAeRunO4CEXD47GxqPw3bmwJ7fr?usp=sharing
5. Create a directory `./data/takekuchi/source` and put downloaded data into three directories `motion/`, `speech/` and `speeche_features/`, separately.

```
.
data
--takekuchi
   --source
      --motion
      --speech
      --speech_features
```

6. split train, dev, and test, `python datasets/takekuchi/data_processing/prepare_data.py`
7. preprocess dataset `python datasets/takekuchi/data_processing/create_vector.py`
8. train model `python main.py wgan takekuchi hparams/wgan/paper_version.json`
9. Evaluate the model on test set (which was not used during training)
   1. Copy saved model checkpoint path (in `results/log_date/chkpt`) to `Infer: pre_trained` inside `hparams/wgan/paper_version.json`
      ```
      "Infer": {
        "pre_trained": "results/log_20210905_2217/chkpt/generator_232k.pt"
      },
      ```
   2. Execute `python main.py wgan takekuchi hparams/wgan/paper_version.json` to obtain the generated motion in 'synthesized/' and the KDE results.

## Visualization
See https://github.com/wubowen416/unity_gesture_visualizer.

## Citation
To cite our work, you can use the following BibTex reference
```
@inproceedings{10.1145/3461615.3485407,
author = {Wu, Bowen and Liu, Chaoran and Ishi, Carlos T. and Ishiguro, Hiroshi},
title = {Probabilistic Human-like Gesture Synthesis from Speech Using GRU-Based WGAN},
year = {2021},
isbn = {9781450384711},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3461615.3485407},
doi = {10.1145/3461615.3485407},
abstract = { Gestures are crucial for increasing the human-likeness of agents and robots to achieve smoother interactions with humans. The realization of an effective system to model human gestures, which are matched with the speech utterances, is necessary to be embedded in these agents. In this work, we propose a GRU-based autoregressive generation model for gesture generation, which is trained with a CNN-based discriminator in an adversarial manner using a WGAN-based learning algorithm. The model is trained to output the rotation angles of the joints in the upper body, and implemented to animate a CG avatar. The motions synthesized by the proposed system are evaluated via an objective measure and a subjective experiment, showing that the proposed model outperforms a baseline model which is trained by a state-of-the-art GAN-based algorithm, using the same dataset. This result reveals that it is essential to develop a stable and robust learning algorithm for training gesture generation models. Our code can be found in https://github.com/wubowen416/gesture-generation.},
booktitle = {Companion Publication of the 2021 International Conference on Multimodal Interaction},
pages = {194–201},
numpages = {8},
keywords = {gesture generation, neural network, social robots, deep learning, generative model},
location = {Montreal, QC, Canada},
series = {ICMI '21 Companion}
}

@inproceedings{wu2021probabilistic,
  title={Probabilistic human-like gesture synthesis from speech using GRU-based WGAN},
  author={Wu, Bowen and Ishi, Carlos and Ishiguro, Hiroshi and others},
  booktitle={GENEA: Generation and Evaluation of Non-verbal Behaviour for Embodied Agents Workshop 2021},
  year={2021}
}
```

## Contact
For any questions, please contact wu.bowen@irl.sys.es.osaka-u.ac.jp
