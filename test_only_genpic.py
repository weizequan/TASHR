import argparse
import numpy as np
import tensorflow as tf
import os
import cv2
import glob
from lite_model import Removal
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', default='/...', type=str,
                    help='The directory of tensorflow checkpoint.')


def data_batch(list1, list2):
    test_dataset = tf.data.Dataset.from_tensor_slices((list1, list2))
    input_size = 512

    def image_fn(path_gt, path_img):
        x = tf.read_file(path_gt)
        x_decode = tf.image.decode_jpeg(x, channels=3)
        gt = tf.image.resize_images(x_decode, [input_size, input_size])
        gt = tf.cast(gt, tf.float32)

        x = tf.read_file(path_img)
        x_decode = tf.image.decode_jpeg(x, channels=3)
        img = tf.image.resize_images(x_decode, [input_size, input_size])
        img = tf.cast(img, tf.float32)
        return gt, img

    test_dataset = test_dataset. \
        repeat(1). \
        map(image_fn). \
        apply(tf.contrib.data.batch_and_drop_remainder(1)). \
        prefetch(1)

    test_gt, test_image = test_dataset.make_one_shot_iterator().get_next()
    return test_gt, test_image


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parser.parse_args()

    config_path = os.path.join('config.yml')
    config = Config(config_path)
    model = Removal(config)

    path_img = '/.../light'
    path_gt = '/.../gt'
    list_gt = list(glob.glob(path_gt + '/*.jpg')) + list(glob.glob(path_gt + '/*.png'))
    list_gt.sort()
    list_img = list(glob.glob(path_img + '/*.jpg')) + list(glob.glob(path_img + '/*.png'))
    list_img.sort()

    gt, images = data_batch(list_gt, list_img)
    images = (images / 255 - 0.5) / 0.5
    gt = (gt / 255 - 0.5) / 0.5
    inputs = images
    # process outputs

    x1, s1 = model.remove_generator(
        images, reuse=False, training=False, name='remove_generator',
        padding='SAME')
    output = x1
    masks = s1


    images = (images + 1) / 2 * 255
    gt = (gt + 1) / 2 * 255
    outputs = (output + 1) / 2 * 255
    masks = tf.concat([masks, masks, masks], axis=3)
    masks=masks*255

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))

        sess.run(assign_ops)
        list20 = []
        list25 = []
        list30 = []
        list35 = []
        listup = []
        n = 0
        maxpsnr = 0.0
        print(len(list_img))
        res_path = '/.../res/'

        # res_path = '../exps/test/new_tmp_notxtloss/'
        if os.path.exists(res_path):
            print("res_path已经存在")
        else:
            os.makedirs(res_path)
        for num in range(0, len(list_img)):
            gted, out, img, mas = sess.run([gt, outputs, images,masks])
            out = out[0][:, :, ::-1].astype(np.uint8)
            mas = mas[0][:, :, ::-1].astype(np.uint8)
            picname_tmp = list_img[num].split('/')[-1]
            picname = picname_tmp.split('_')[0]

            percent = '%.2f' % (1.0 * num / len(list_img) * 100)
            print('\rcomplete percent:' + str(percent) + '%', end='')

            cv2.imwrite(res_path + '{}_res.png'.format(str(picname)), out)
            # cv2.imwrite(res_path + '{}_mask.png'.format(str(picname)), mas)
        print('end!')

