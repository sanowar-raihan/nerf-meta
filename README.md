# NeRF Meta Learning With PyTorch
**nerf-meta** is a PyTorch re-implementation of NeRF experiments from the paper ["Learned Initializations for Optimizing Coordinate-Based Neural Representations"](https://arxiv.org/abs/2012.02189). Simply by initializing NeRF with meta-learned weights, we can achieve:

* [Photo Tourism](#photo-tourism)
* [View Synthesis from Single Image](#view-synthesis-from-single-image)

Be sure to check out the original resources from the authors:

* **[Paper]**: [https://arxiv.org/abs/2012.02189](https://arxiv.org/abs/2012.02189)
* **[Original Project Page]**: [https://www.matthewtancik.com/learnit](https://www.matthewtancik.com/learnit)  
* **[Official JAX Implementation]**: [https://github.com/tancik/learnit](https://github.com/tancik/learnit)  
* **[Twitter Summary]**: [https://twitter.com/BenMildenhall/status/1335290097390895104](https://twitter.com/BenMildenhall/status/1335290097390895104)

### Environment

* Python 3.8
* PyTorch 1.8
* NumPy, imageio, imageio-ffmpeg

## Photo Tourism

Starting from a meta-initialized NeRF, we can interpolate between camera pose, focal length, aspect ratio and scene appearance. The videos below are generated with a 5 layer only NeRF, trained for ~100k iterations.

https://user-images.githubusercontent.com/45260868/117478335-d50ce980-af80-11eb-8e4f-dadb1a209600.mp4

https://user-images.githubusercontent.com/45260868/117478378-e35b0580-af80-11eb-94c9-50eafbc560c5.mp4

**Data**
* Get image collection of different landmarks from [image-matching-challenge](https://www.cs.ubc.ca/~kmyi/imw2020/data.html)
* Get poses and bounds from [learnit google drive](https://drive.google.com/drive/folders/1SVHKRQXiRb98q4KHVEbj8eoWxjNS2QLW)

**Train and Evaluate**
1. Train NeRF on a single landmark scene using [Reptile](https://arxiv.org/abs/1803.02999) meta-learning:
    ```shell
    python tourism_train.py --config ./configs/tourism/$landmark.json
    ```
2. Test Photo Tourism performance and generate an interpolation video of the landmark:
    ```shell
    python tourism_test.py --config ./configs/tourism/$landmark.json --weight-path $meta_weight.pth
    ```

## View Synthesis from Single Image

Given a single input view, meta-initialized NeRF can generate a 360-degree video. The following ShapeNet video is generated with a class-specific NeRF (5 layers deep), trained for ~100k iterations.

https://user-images.githubusercontent.com/45260868/117478568-26b57400-af81-11eb-8770-4716a680c61b.mp4

**Data**
* Get ShapeNet Data and splits files from [learnit google drive](https://drive.google.com/drive/folders/1SVHKRQXiRb98q4KHVEbj8eoWxjNS2QLW)

**Train and Evaluate**
1. Train NeRF on a particular ShapeNet class using [Reptile](https://arxiv.org/abs/1803.02999) meta-learning:
    ```shell
    python shapenet_train.py --config ./configs/shapenet/$shape.json
    ```
2. Optimize the meta-trained model on a single view and test on other held-out views. It also generates a 360 video for each test object:
    ```shell
    python shapenet_test.py --config ./configs/shapenet/$shape.json --weight-path $meta_weight.pth
    ```

## Acknowledgments
I referenced several open-source NeRF and Meta-Learning code base for this implementation. Specifically, I borrowed/modified code from the following repositories:

* [learnit](https://github.com/tancik/learnit)
* [JAX NeRF](https://github.com/google-research/google-research/tree/master/jaxnerf)
* [NeRF PyTorch Lightning](https://github.com/kwea123/nerf_pl)
* [learn2learn](https://github.com/learnables/learn2learn)

Thanks to the authors for releasing their code.
