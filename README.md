# revisiting-sepconv
This is a reference implementation of Revisiting Adaptive Convolutions for Video Frame Interpolation [1] using PyTorch. Given two frames, it will make use of [adaptive convolution](http://sniklaus.com/papers/adaconv) [2] in a separable manner [3] to interpolate the intermediate frame. Should you be making use of our work, please cite our paper [1].

<a href="https://arxiv.org/abs/2011.01280" rel="Paper"><img src="http://content.sniklaus.com/resepconv/paper.jpg" alt="Paper" width="100%"></a>

For the original SepConv, see: https://github.com/sniklaus/sepconv-slomo
<br />
For softmax splatting, please see: https://github.com/sniklaus/softmax-splatting

## setup
The separable convolution layer is implemented in CUDA using CuPy, which is why CuPy is a required dependency. It can be installed using `pip install cupy` or alternatively using one of the provided [binary packages](https://docs.cupy.dev/en/stable/install.html#installing-cupy) as outlined in the CuPy repository.

If you plan to process videos, then please also make sure to have `pip install moviepy` installed.

## usage
To run it on your own pair of frames, use the following command.

```
python run.py --model paper --one ./images/one.png --two ./images/two.png --out ./out.png
```

To run in on a video, use the following command.

```
python run.py --model paper --video ./videos/car-turn.mp4 --out ./out.mp4
```

For a quick benchmark using examples from the Middlebury benchmark for optical flow, run `python benchmark.py`. You can use it to easily verify that the provided implementation runs as expected.

## video
<a href="http://content.sniklaus.com/resepconv/video.mp4" rel="Video"><img src="http://content.sniklaus.com/resepconv/video.jpg" alt="Video" width="100%"></a>

## license
Please refer to the appropriate file within this repository.

## references
```
[1]  @inproceedings{Niklaus_WACV_2021,
         author = {Simon Niklaus and Long Mai and Oliver Wang},
         title = {Revisiting Adaptive Convolutions for Video Frame Interpolation},
         booktitle = {IEEE Winter Conference on Applications of Computer Vision},
         year = {2021}
     }
```

```
[2]  @inproceedings{Niklaus_ICCV_2017,
         author = {Simon Niklaus and Long Mai and Feng Liu},
         title = {Video Frame Interpolation via Adaptive Separable Convolution},
         booktitle = {IEEE International Conference on Computer Vision},
         year = {2017}
     }
```

```
[3]  @inproceedings{Niklaus_CVPR_2017,
         author = {Simon Niklaus and Long Mai and Feng Liu},
         title = {Video Frame Interpolation via Adaptive Convolution},
         booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
         year = {2017}
     }
```