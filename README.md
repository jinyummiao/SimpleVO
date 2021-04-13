# SimpleVO
A Python toy demo about visual odometry. I simply track features between current image and last keyframe and then perform triangulation. The demo does not have local BA optimization and is used to evaluate feature tracking/matching algorithms in the context of SLAM. The relative motion between current frame and last keyframe is estimated by OpenCV functions. This project is inspirsed and based on [Python-VO](https://github.com/Shiaoming/Python-VO) and [monocular_slam](https://github.com/YunYang1994/openwork/tree/main/monocular_slam). 

Note that the project is research code. The author is not responsible for any errors it may contain. Use it at your own risk!

## Requirements
PyTorch, OpenCV, [Pangolin](https://github.com/YunYang1994/pangolin), PyOpenGL ...

## Usage
You need to download the outdoor pre-trained parameters of SuperGlue and save them in the /src/matcher/SuperGlue/weights/

```
python demo.py --config configs/params/kitti_**_**.yaml
```


