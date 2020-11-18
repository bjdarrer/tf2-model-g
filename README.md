# tf2-model-g
Model G implemented in Tensorflow v2

## Installation
This code is intented to run on Google Cloud on the Deep Learning VM available in the Marketplace.
To install any dependencies not already included in the VM run
```bash
pip3 install -r requirements.txt
```

## Local development

### Requirements
- python3
```bash
sudo apt install python3
```
- pip3
```bash
sudo apt update -y
sudo apt install python3-pip -y
```
- [Tensorflow 2](https://www.tensorflow.org/install)
```bash
pip3 install tensorflow
```
- Additional dependencies
```bash
pip3 install -r requirements.txt
```
run code example:
```bash
python3 render_video.py ~/tf2-model-g/nucleation_and_motion_in_fluid_2D.mp4 --params params/nucleation_and_motion_in_fluid_2D.yaml
```
run with plotting facility
```bash
pip3 install matplotlib
```