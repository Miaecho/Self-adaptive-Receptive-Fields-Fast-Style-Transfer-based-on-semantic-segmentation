import argparse
import os
import shutil
import tensorflow as tf
from model import Model
from utils import mkdir_if_not_exists
from skimage import util, io
from PIL import Image
import numpy as np
# import time
#
# start = time.time()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_parser():
    parser = argparse.ArgumentParser()

    # Input Options
    parser.add_argument('--train_path', type=str, dest='train_path', help='training images folder path',
                        default="./data/MSCOCO")
    parser.add_argument('--content_path', type=str, dest="content_path", help='content image path',
                        default='./examples/content/1.png')
    parser.add_argument('--style_path', type=str, dest='style_path', help='style image path',
                        default='./examples/style/1.png')

    parser.add_argument('--batch_size', type=int, dest='batch_size',
                        help='batch size (default %(default)s)', default=1)
    parser.add_argument('--learning_rate', type=float, dest='learning_rate',
                        help='learning rate (default %(default)s)', default=1e-3)

    parser.add_argument('--max_iter', type=int, dest='max_iter',
                        help='max iterations (default %(default)s)', default=1e4)
    parser.add_argument('--iter_print', type=int, dest='iter_print',
                        help='per iterations display (default %(default)s)', default=1e3)
    parser.add_argument('--checkpoint_iter', type=int, dest='cp_iter',
                        help='per iterations save (default %(default)s)', default=5e3)

    # Weight Options
    parser.add_argument('--content_weight', type=float, dest="content_weight",
                        help='content weight (default %(default)s)', default=50)
    parser.add_argument('--style_weight', type=float, dest="style_weight",
                        help='style weight (default %(default)s)', default=30)
    parser.add_argument('--tv_weight', type=float, dest="tv_weight",
                        help="total variation regularization weight (default %(default)s)", default=2e2)

    # Finetune Options
    parser.add_argument('--continue_train', type=bool, dest='continue_train', default=False)

    # Others
    parser.add_argument('--noise', type=str, dest="noise", help="options: salt_and_pepper, poisson, gaussian, no "
                                                                "(default %(default)s)", default='gaussian')
    parser.add_argument('--noise_amount', type=float, dest="noise_amount",
                        help="noise amount (default %(default)s)", default=8e-4)
    # seg
    parser.add_argument("--content_seg_path", dest='content_seg_path', default='./examples/segmentation/c1.png',
                        help="content segmentation image path")
    parser.add_argument("--style_seg_path", dest='style_seg_path', default='./examples/segmentation/s1.png',
                        help="style segmentation image path")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)
    train_model = Model(sess, args)

    image = Image.open(args.content_path)
    img = np.array(image)
    amount = args.noise_amount

    if args.noise == 'salt_and_pepper':
        img = util.random_noise(img, mode='s&p', amount=amount)
    elif args.noise == 'poisson':
        img = util.random_noise(img, mode='poisson')
    elif args.noise == 'gaussian':
        img = util.random_noise(img, var=amount ** 2, mean=0.05)
    elif args.noise == 'no':
        pass

    style_image_basename = os.path.basename(args.content_path)
    style_image_basename = style_image_basename[:style_image_basename.find(".")]

    if args.noise != 'no':
        n = (style_image_basename + "noise")
        io.imsave("./examples/content/{}.png".format(n), img)
        args.content_path = ("./examples/content/{}.png".format(n))

    args.checkpoint_dir = os.path.join("./examples/checkpoint", style_image_basename)
    args.serial = os.path.join("./examples/serial", style_image_basename)

    print("[*] Checkpoint Directory: {}".format(args.checkpoint_dir))
    print("[*] Serial Directory: {}".format(args.serial))
    mkdir_if_not_exists(args.serial, args.checkpoint_dir)

    if args.continue_train:
        train_model.finetune_model(args)
    else:
        train_model.train(args)

    # elaspsed = (time.time() - start)
    # print 'time used:', elaspsed


if __name__ == "__main__":
    main()
