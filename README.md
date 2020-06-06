Self-adaptive-Receptive-Fields-Fast-Style-Transfer-based-on-semantic-segmentation

## Getting Started

Implemented and tested on Ubuntu 14.04 with Python 2.7 and Tensorflow 1.4.1.

### Dependencies
* [Tensorflow](https://www.tensorflow.org/)
* [Numpy](www.numpy.org/)
* [Pillow](https://pypi.python.org/pypi/Pillow/)
* [Scipy](https://www.scipy.org/)

### Download pre-trained VGG-19 model
The VGG-19 model of tensorflow is adopted from [VGG Tensorflow](https://github.com/machrisaa/tensorflow-vgg) with few modifications on the class interface. The VGG-19 model weights is stored as .npy file and could be download from [Google Drive](https://drive.google.com/file/d/0BxvKyd83BJjYY01PYi1XQjB5R0E/view?usp=sharing) or [BaiduYun Pan](https://pan.baidu.com/s/1o9weflK). After downloading, copy the weight file to the **/vgg19** directory

## Basic Usage
### Train the network
Use train.py to train a new stroke controllable style transfer network. Run `python train.py -h` to view all the possible parameters. The dataset used for training is MSCOCO train 2014 and could be download from [here](http://cocodataset.org/#download), or you can use a random selected 2k images from MSCOCO (download from [here](https://drive.google.com/file/d/1ph85_1YgApUMD0YkGKZ8EOU4xyNoTJYo/view?usp=sharing)) for quick setup. Example usage:

```
$ python train.py \
    --style /path/to/style_image.jpg \
    --train_path /path/to/MSCOCO_dataset \
    --sample_path /path/to/content_image.jpg
    --style_seg_path /path/to/style_seg.jpg
    --content_seg_path /path/to/content_seg.jpg
    --batch_size 1
```

### Freeze model
Use pack_model.py to freeze the saved checkpoint. Run `python pack_model.py -h` to view all parameter. Example usage:

```
$ python pack_model.py \
    --checkpoint_dir ./examples/checkpoint/some_style \
    --output ./examples/model/some_style.pb
```

We also provide some pre-trained style model for fast forwarding, which is stored under `./examples/model/pre-trained/`.

### Inference
Use inference_style_transfer.py to inference the content image based on the freezed style model. Set `--interp N` to enable interpolation inference where `N` is the number of the continuous stroke results.

```
$ python inference_style_transfer.py \
    --model ./examples/model/some_style.pb \
    --serial ./examples/serial/default/ \
    --content ./examples/content/some_content.jpg
```

For CPU-users, please set `os.environ["CUDA_VISIBLE_DEVICES"]=""` in the source code.

