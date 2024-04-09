This is a repo for unguided navigation using LLM and segmentation models on a robot.

Installation instruction:

Create conda env:
```bash
conda create -n grounding_dino
conda activate grounding_dino
pip install rewuirements.txt

```

download the three weight files here:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth
```
Install ros-numpy:
```bash
sudo apt-get install ros-noetic-ros-numpy
```
