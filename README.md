# ChaChaBench

ChaChaBench is a diagnostic benchmark designed to evaluate the ability of Vision-Language Models (VLMs) to understand and reason about camera motion. It provides a suite of tasks and datasets that probe how well models can interpret, describe, and answer questions about dynamic camera movements in visual scenes.

## Features
- Diverse set of camera motion scenarios (e.g., pan, tilt, zoom, orbit)
- Task suite for both recognition and reasoning about camera motion
- Compatible with popular VLMs and evaluation pipelines

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/robertocarlosjuan/ChaChaBench.git
   cd ChaChaBench
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Example usage for evaluating a model:
```bash
python main.py --model_path <model_path>
```
