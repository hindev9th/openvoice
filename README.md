# This repo makes some changes:
## Add script for finetune: 
### To finetune:
- Add directories: `mkdir training_data reference_voices finetuned_model`
- Please please audio files for training in the training_data folder and reference_voices folder, files can be .wav for high quality
- File .wav must be 2-20s long, no noise, clear, high quality, single speaker
- The finetune script will use the base checkpoint.pt model to generate the output then compare this output with the referencing voice.
- The idea is that if we can provide more diverse voices, the model will learn better
- Finetune script:
  `
  python3 finetune_tone_converter.py \
    --config_path config.json \
    --model_path checkpoint.pth \
    --data_dir training_data \
    --reference_audio reference_voices \
    --output_dir finetuned_model \
    --epochs 100 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --device cuda
  `
### To test the finetuned model:
- Run the script: `python3 inference_pt.py --config_path config.json --model_path finetuned_model/final_model.pth --input_audio biden.wav --reference_audio trump.wav --output_dir outputs --debug`
  
## ONNX export and inference: 
### How to use it?
- For export: `python3 export_onnx.py --model_path checkpoint.pth --config_path config.json --output_path tone_converter.onnx`

- For inference: `python3 inference.py --config_path config.json --input_audio trump.wav --reference_audio biden.wav --output_dir outputs --device cpu `

<div align="center">
  <div>&nbsp;</div>
  <img src="resources/openvoicelogo.jpg" width="400"/> 

[Paper](https://arxiv.org/abs/2312.01479) |
[Website](https://research.myshell.ai/open-voice) <br> <br>
<a href="https://trendshift.io/repositories/6161" target="_blank"><img src="https://trendshift.io/api/badge/repositories/6161" alt="myshell-ai%2FOpenVoice | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

## Introduction

### OpenVoice V1

As we detailed in our [paper](https://arxiv.org/abs/2312.01479) and [website](https://research.myshell.ai/open-voice), the advantages of OpenVoice are three-fold:

**1. Accurate Tone Color Cloning.**
OpenVoice can accurately clone the reference tone color and generate speech in multiple languages and accents.

**2. Flexible Voice Style Control.**
OpenVoice enables granular control over voice styles, such as emotion and accent, as well as other style parameters including rhythm, pauses, and intonation. 

**3. Zero-shot Cross-lingual Voice Cloning.**
Neither of the language of the generated speech nor the language of the reference speech needs to be presented in the massive-speaker multi-lingual training dataset.

### OpenVoice V2

In April 2024, we released OpenVoice V2, which includes all features in V1 and has:

**1. Better Audio Quality.**
OpenVoice V2 adopts a different training strategy that delivers better audio quality.

**2. Native Multi-lingual Support.**
English, Spanish, French, Chinese, Japanese and Korean are natively supported in OpenVoice V2.

**3. Free Commercial Use.**
Starting from April 2024, both V2 and V1 are released under MIT License. Free for commercial use.

[Video](https://github.com/myshell-ai/OpenVoice/assets/40556743/3cba936f-82bf-476c-9e52-09f0f417bb2f)

OpenVoice has been powering the instant voice cloning capability of [myshell.ai](https://app.myshell.ai/explore) since May 2023. Until Nov 2023, the voice cloning model has been used tens of millions of times by users worldwide, and witnessed the explosive user growth on the platform.

## Main Contributors

- [Zengyi Qin](https://www.qinzy.tech) at MIT
- [Wenliang Zhao](https://wl-zhao.github.io) at Tsinghua University
- [Xumin Yu](https://yuxumin.github.io) at Tsinghua University
- [Ethan Sun](https://twitter.com/ethan_myshell) at MyShell

## How to Use
Please see [usage](docs/USAGE.md) for detailed instructions.

## Common Issues

Please see [QA](docs/QA.md) for common questions and answers. We will regularly update the question and answer list.

## Citation
```
@article{qin2023openvoice,
  title={OpenVoice: Versatile Instant Voice Cloning},
  author={Qin, Zengyi and Zhao, Wenliang and Yu, Xumin and Sun, Xin},
  journal={arXiv preprint arXiv:2312.01479},
  year={2023}
}
```

## License
OpenVoice V1 and V2 are MIT Licensed. Free for both commercial and research use.

## Acknowledgements
This implementation is based on several excellent projects, [TTS](https://github.com/coqui-ai/TTS), [VITS](https://github.com/jaywalnut310/vits), and [VITS2](https://github.com/daniilrobnikov/vits2). Thanks for their awesome work!
