# **Y**ou **O**nly **L**ook **A**t **C**oefficien**T**s



# Installation
 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/dbolya/yolact.git
   cd yolact
   ```
 - Set up the environment using one of the following methods:
   - Using [Anaconda](https://www.anaconda.com/distribution/)
     - Run `conda env create -f environment.yml`
   - Manually with pip
     - Set up a Python3 environment (e.g., using virtenv).
     - Install [Pytorch](http://pytorch.org/) 1.0.1 (or higher) and TorchVision.
     - Install some other packages:
       ```Shell
       # Cython needs to be installed before pycocotools
       pip install cython
       pip install opencv-python pillow pycocotools matplotlib 
       ```
- Download the darknet-pretrained backbone from [here](https://drive.google.com/file/d/17Y431j4sagFpSReuPNoFcj9h7azDTZFf/view?usp=sharing) and the pretrained weights from [here]  and put them in `yolact/weights`.

# Inference
 * Images
```
## Images
```Shell
# Process and sisplay one image
python eval.py \
--trained_model=weights/yolact_base_54_800000.pth \
--score_threshold=0.9 \
--top_k=15 \
--image=my_image.png

# Process an image and save it to another file.
python eval.py \
--trained_model=weights/yolact_base_54_800000.pth \
--score_threshold=0.9 \
--top_k=15 \
--image=input_image.png:output_image.png

# Process a whole folder of images.
python eval.py \
--trained_model=weights/yolact_base_54_800000.pth \
--score_threshold=0.9 \
--top_k=15 \
--images=path/to/input/folder:path/to/output/folder
```
* Videos

```Shell
# Display a video in real-time. "--video_multiframe" will process that many frames at once for improved performance.
# If you want, use "--display_fps" to draw the FPS directly on the frame.
python eval.py \
--trained_model=weights/yolact_base_54_800000.pth \
--score_threshold=0.15 \
--top_k=15 \
--video_multiframe=4 \
--video=my_video.mp4

# Process a video and save it to another file. 
python eval.py \
--trained_model=weights/yolact_base_54_800000.pth \
--score_threshold=0.15 \
--top_k=15 \
--video_multiframe=4 \
--video=input_video.mp4:output_video.mp4
```


# Training
* All weights are saved in the `yolact/weights` directory by default with the file name `<config>_<epoch>_<iter>.pth`.
* The configuration used in this project uses darknet as a backbone and focal loss as a loss function.
```Shell
# Trains using the config in data/config.py file.
python train.py --config=yolact_darknet53_config --batch_size=16
```

# Evaluating
* To get AP@[0.5,0.95] for bounding boxes and masks
```shell
python yolact/eval.py \
--trained_model=yolact/weights/yolact/weights/checkpoint_name.pth \
--config=yolact_darknet53_config 
```
* To get coco-style evaluation metrics for bounding boxes and masks:
    * Generate predictions in json format (this will create 2 files `yolact/results/bbox_detections.json` and `yolact/results/mask_detections.json`)
    
    ```shell
    python yolact/eval.py \
    --trained_model=yolact/weights/yolact/weights/checkpoint_name.pth  \
    --config=yolact_base_config \
    --output_coco_json \
    --dataset=wildlife_valdev_dataset 
    ```
    * Get coco-style evaluation
    ```shell
    python yolact/run_coco_eval.py \
    --bbox_det_file=yolact/results/bbox_detections.json \
    --mask_det_file=yolact/results/mask_detections.json \
    --gt_ann_file=dataset_v5/val/annotations.json \
    --eval_type='both'
    ```
