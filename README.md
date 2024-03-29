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
@inproceedings{wu2021probabilistic,
  title={Probabilistic human-like gesture synthesis from speech using GRU-based WGAN},
  author={Wu, Bowen and Ishi, Carlos and Ishiguro, Hiroshi and others},
  booktitle={GENEA: Generation and Evaluation of Non-verbal Behaviour for Embodied Agents Workshop 2021},
  year={2021}
}
```

## Contact
For any questions, feel free to open an issue.
