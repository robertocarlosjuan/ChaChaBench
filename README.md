# ChaChaBench

**Understanding the observer’s motion is as essential as perceiving the scene itself.**
Despite advances in video-language models (VLMs), their ability to recognize the camera’s movement—independent of scene content—remains largely untested.

We evaluate whether VLMs can accurately identify **basic egocentric camera motions**—such as *move forward*, *pan left*, or *tilt up*—from short, single-motion videos. To isolate this core ability, we generate 1,000+ minimalistic, noise-free clips in **Omnigibson**, avoiding any object motion, lighting artifacts, or compositional complexity.

Most VLMs are pre-trained on scene-centric captions. But reasoning about space, perspective, and viewer motion is **foundational** for tasks like navigation, tracking, and embodiment. Our benchmark directly probes this capacity.

* Gemini-2.0 & Gemini-2.5 (Flash)
* Qwen2.5-VL (7B / 32B / 72B, Instruct)
* NVILA-15B & LongVILA
* CameraBench (SFT)

### Dataset Format

* 12-class taxonomy: {move/tilt/pan/roll in 6DoF directions}
* Videos + JSON annotations
* Compatible with any vision-language pipeline

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
Load data from [HuggingFace](https://huggingface.co/datasets/carihe/ChaChaBench) into `data/`

## Evaluation

To evaluate on ChaChaBench, run:

```eval
python main.py --model_path <model_path>
```

## Results

| Model                   | Avg F1 (%) |
| ----------------------- | ---------- | 
| Human (reference)       | 97.2       | 
| Gemini-2.5-Flash (best) | 35.6       | 
| Qwen2.5-VL-72B-Instruct | 25.8       | 

* All models severely underperform on **move backward** (F1 ≈ 0–7%)
* Many models conflate **rotation** (e.g., pan/tilt) with **translation** (e.g., move left/right)
* Bias toward forward motion with **low precision** in the forward class, showing poor class discrimination
* **Roll**, confuses even larger models

