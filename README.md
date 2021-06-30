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

# Evaluation


To evalute the model, put the corresponding weights file in the `./weights` directory and run one of the following commands. The name of each config is everything before the numbers in the file name (e.g., `yolact_base` for `yolact_base_54_800000.pth`).
## Quantitative Results on COCO
```Shell
# Quantitatively evaluate a trained model on the entire validation set. Make sure you have COCO downloaded as above.
# This should get 29.92 validation mask mAP last time I checked.
python eval.py --trained_model=weights/yolact_base_54_800000.pth

# Output a COCOEval json to submit to the website or to use the run_coco_eval.py script.
# This command will create './results/bbox_detections.json' and './results/mask_detections.json' for detection and instance segmentation respectively.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --output_coco_json

# You can run COCOEval on the files created in the previous command. The performance should match my implementation in eval.py.
python run_coco_eval.py

# To output a coco json file for test-dev, make sure you have test-dev downloaded from above and go
python eval.py --trained_model=weights/yolact_base_54_800000.pth --output_coco_json --dataset=coco2017_testdev_dataset
```
## Qualitative Results on COCO
```Shell
# Display qualitative results on COCO. From here on I'll use a confidence threshold of 0.15.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --display
```
## Benchmarking on COCO
```Shell
# Run just the raw model on the first 1k images of the validation set
python eval.py --trained_model=weights/yolact_base_54_800000.pth --benchmark --max_images=1000
```
## Images
```Shell
# Display qualitative results on the specified image.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=my_image.png

# Process an image and save it to another file.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=input_image.png:output_image.png

# Process a whole folder of images.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --images=path/to/input/folder:path/to/output/folder
```
## Video
```Shell
# Display a video in real-time. "--video_multiframe" will process that many frames at once for improved performance.
# If you want, use "--display_fps" to draw the FPS directly on the frame.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=my_video.mp4

# Display a webcam feed in real-time. If you have multiple webcams pass the index of the webcam you want instead of 0.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=0

# Process a video and save it to another file. This uses the same pipeline as the ones above now, so it's fast!
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=input_video.mp4:output_video.mp4
```
As you can tell, `eval.py` can do a ton of stuff. Run the `--help` command to see everything it can do.
```Shell
python eval.py --help
```


# Training
By default, we train on COCO. Make sure to download the entire dataset using the commands above.
- To train, download the darknet-pretrained model from [here](https://drive.google.com/file/d/17Y431j4sagFpSReuPNoFcj9h7azDTZFf/view?usp=sharing). and put it in `./weights`.
- All weights are saved in the `./weights` directory by default with the file name `<config>_<epoch>_<iter>.pth`.
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
