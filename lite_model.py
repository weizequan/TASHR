import sys
import os
from lite_network import *
from lite_metrics import *
from lite_loss import *
from tensorflow.contrib.framework.python.ops import arg_scope
import tensorflow as tf
from ocr.ctpn.lib.networks.VGGnet_test import VGGnet_test
import densenet
import keys
from imp import reload

class Removal(object):
    def __init__(self, config):
        self.config = config
        self.res_num = config.RES_NUM
        self.base_channel = config.BASE_CHANNEL
        self.sample_num = config.SAMPLE_NUM
        self.model_name = 'removal'
        self.rate=config.RATE
        self.netDet = VGGnet_test(trainable=False)

        self.gen_optimizer = tf.train.AdamOptimizer(
            learning_rate=float(config.LR),
            beta1=float(config.BETA1),
            beta2=float(config.BETA2)
        )
        self.dis_optimizer = tf.train.AdamOptimizer(
            learning_rate=float(config.LR),
            beta1=float(config.BETA1),
            beta2=float(config.BETA2)
        )

    def get_nclass(self):
        characters = keys.alphabet[:]
        characters = characters[1:] + u'卍'
        nclass = len(characters)

    def build_whole_model(self, batch_data, is_training=True):
        batch_predicted,batch_gt,gen_loss,dis_loss=self.remove_model(batch_data, training=is_training)
        outputs_merged = (batch_predicted + 1) / 2 * 255
        gt = (batch_gt + 1) / 2 * 255
        _, psnr = mean_psnr(gt, outputs_merged)
        _, ssim = mean_ssim(gt, outputs_merged)
        tf.summary.scalar('train/psnr', psnr)
        tf.summary.scalar('train/ssim', ssim)
        tf.summary.scalar('train_loss/gen_loss', gen_loss)
        tf.summary.scalar('train_loss/dis_loss', dis_loss)
        return gen_loss,dis_loss,psnr,ssim
    def build_validation_model(self, batch_data, reuse=True, is_training=False):  
        batch_batch = batch_data
        batch_width = int(batch_batch.get_shape().as_list()[2]/3)
        batch_gt = batch_batch[:, :, :batch_width,:] / 127.5 - 1.
        batch_img = batch_batch[:, :, batch_width:batch_width*2,:] / 127.5 - 1.
        batch_mask = tf.cast(batch_batch[:, :, batch_width*2:batch_width*3,0:1] > 127.5, tf.float32)
        batch_incomplete = batch_img

        print('val_batch_batch.shape',batch_batch.shape)
        print('val_batch_gt.shape',batch_gt.shape)
        print('val_batch_img.shape',batch_img.shape)
        print('val_batch_mask.shape',batch_mask.shape)

        # process outputs
        x1, s1 = self.remove_generator(
            batch_img, reuse=reuse, training=is_training,name=self.model_name + '_generator',
            padding='SAME')
        batch_predicted = x1
        batch_complete = batch_predicted*batch_mask + batch_img*(1.-batch_mask)
        batch_complete = batch_predicted
        viz_img = tf.concat([x1, 
                tf.tile(batch_mask,[1,1,1,3]), tf.tile(s1,[1,1,1,3])], axis=2)
        images_summary(viz_img,'x1_masks_text_s1', 10)
        

        _, psnr = mean_psnr((batch_img+1.)*127.5, (batch_complete+1.)*127.5)
        _, ssim = mean_ssim((batch_img+1.)*127.5, (batch_complete+1.)*127.5)
        return psnr,ssim

    def build_optim(self, gen_loss, dis_loss):
        g_vars = tf.get_collection( 
            tf.GraphKeys.TRAINABLE_VARIABLES, self.model_name + '_generator')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        g_gradient = self.gen_optimizer.compute_gradients(gen_loss, var_list=g_vars)
        d_gradient = self.dis_optimizer.compute_gradients(dis_loss, var_list=d_vars)
        return self.gen_optimizer.apply_gradients(g_gradient), self.dis_optimizer.apply_gradients(d_gradient)

    def remove_model(self, batch_data, training=True,reuse=False):
        batch_batch = batch_data
        batch_width = int(batch_batch.get_shape().as_list()[2]/3)
        batch_gt = batch_batch[:, :, :batch_width,:] / 127.5 - 1.
        batch_img = batch_batch[:, :, batch_width:batch_width*2,:] / 127.5 - 1.
        batch_mask = tf.cast(batch_batch[:, :, batch_width*2:batch_width*3,0:1] > 127.5, tf.float32)
        batch_incomplete = batch_img

        # process outputs
        batch_img = tf.add(batch_img, 0, name='input_image')
        batch_mask = tf.add(batch_mask, 0, name='input_mask')
        x1, s1 = self.remove_generator(
            batch_img, reuse=reuse, training=training,name=self.model_name + '_generator',
            padding='SAME')
        batch_predicted = x1 

        self.netDet.setup(data=batch_predicted, im_info=tf.stack([tf.constant((256,256,1))]*4), name='text_detector')
        det_output_f1 = self.netDet.get_output('conv5_1')
        det_output_f2 = self.netDet.get_output('conv5_2')
        det_output_f3 = self.netDet.get_output('conv5_3')
        self.netDet.setup(data=batch_gt, im_info=tf.stack([tf.constant((256,256,1))]*4), name='text_detector', reuse=True)
        det_gt_f1 = self.netDet.get_output('conv5_1')
        det_gt_f2 = self.netDet.get_output('conv5_2')
        det_gt_f3 = self.netDet.get_output('conv5_3')
    
        reload(densenet)
        disc_output = densenet.dense_cnn(batch_predicted, 5000)
        disc_gt = densenet.dense_cnn(batch_gt, 5000)

        if training:
            losses = {}
            losses['netD_loss'] = tf.reduce_mean(tf.abs(batch_mask - s1))

            loss_pw = tf.reduce_mean(tf.abs(batch_gt - x1))
            loss_tv = tv_loss(x1)
            losses['pixel_loss'] = 5 * loss_pw + 0.1 *loss_tv

            loss_content = perceptual_loss(x1, batch_gt)
            loss_style = style_loss(x1, batch_gt)
            losses['feature_loss'] = 0.05 * loss_content + 120 * loss_style

        # # seperate gan
            batch_pos_feature = self.build_sngan_discriminator(batch_gt, training=training, reuse=reuse)
            batch_neg_feature = self.build_sngan_discriminator(batch_predicted, training=training, reuse=tf.AUTO_REUSE)
            loss_discriminator, loss_generator = hinge_gan_loss(batch_pos_feature, batch_neg_feature)
            losses['gan_loss'] = 0.01 * loss_generator
            losses['d_loss'] = loss_discriminator

            loss_txtD = 0.0
            loss_txtD += tf.reduce_mean(tf.abs(det_output_f1 - det_gt_f1))
            loss_txtD += tf.reduce_mean(tf.abs(det_output_f2 - det_gt_f2))
            loss_txtD += tf.reduce_mean(tf.abs(det_output_f3 - det_gt_f3))
            loss_txtR = tf.reduce_mean(tf.abs(disc_output - disc_gt))
            losses['txt_loss'] = loss_txtD + loss_txtR

            losses['g_loss'] = 10 * losses['netD_loss'] + losses['pixel_loss'] + losses['feature_loss'] + losses['gan_loss'] + losses['txt_loss']


        
            viz_img = [batch_gt, batch_img, tf.tile(s1,[1,1,1,3]), x1]

            images_summary(
                tf.concat(viz_img, axis=2),
                'raw_incomplete_predicted_complete', 10)
            return batch_predicted, batch_gt,losses['g_loss'],losses['d_loss']

        else:
            return batch_predicted, batch_gt,batch_img

    def remove_generator(self, x, reuse=False,
                        training=True, padding='SAME', name='generator'):
        """Inpaint network.

        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image 
        """
        
        xin = x
        
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        x = tf.concat([x, ones_x], axis=3)

        # two stage network
        cnum = 32
        with tf.variable_scope(name, reuse=reuse), \
                arg_scope([gen_conv, gen_deconv],
                        training=training, padding=padding):

            # x mask的大小print
            t1_conv1 = gen_conv(x, cnum//2, 3, 1, name='t1conv1')
            t1_conv2 = gen_conv(t1_conv1, cnum//2, 3, 1, name='t1conv2')
            t1_conv3 = gen_conv(t1_conv2, cnum, 3, 2, name='t1conv3_128')
            t1_conv4 = gen_conv(t1_conv3, cnum, 3, 1, name='t1conv4')
            t1_conv5 = gen_conv(t1_conv4, cnum, 3, 1, name='t1conv5')
            t1_conv6 = gen_conv(t1_conv5, 2*cnum, 3, 2, name='t1conv6_64')
            t1_conv7 = gen_conv(t1_conv6, 2*cnum, 3, 1, name='t1conv7')
            t1_conv8 = gen_conv(t1_conv7, 2*cnum, 3, 1, name='t1conv8')
            t1_conv9 = gen_conv(t1_conv8, 4*cnum, 3, 2, name='t1conv9_32')
            t1_conv10 = gen_conv(t1_conv9, 4*cnum, 3, 1, name='t1conv10')
            t1_conv11 = gen_deconv(t1_conv10, 2*cnum, name='t1conv11_64')
            t1_conv11 = tf.concat([t1_conv8, t1_conv11], axis=3)
            t1_conv12 = gen_conv(t1_conv11, 2*cnum, 3, 1, name='t1conv12')
            t1_conv13 = gen_conv(t1_conv12, 2*cnum, 3, 1, name='t1conv13')
            t1_conv14 = gen_conv(t1_conv13, 2*cnum, 3, 1, name='t1conv14')
            t1_conv15 = gen_deconv(t1_conv14, cnum, name='t1conv15_128')
            t1_conv15 = tf.concat([t1_conv5, t1_conv15], axis=3)
            t1_conv16 = gen_conv(t1_conv15, cnum, 3, 1, name='t1conv16')
            t1_conv17 = gen_conv(t1_conv16, cnum, 3, 1, name='t1conv17')
            t1_conv18 = gen_conv(t1_conv17, cnum, 3, 1, name='t1conv18')
            t1_conv19 = gen_deconv(t1_conv18, cnum//2, name='t1conv19_256')
            t1_conv19 = tf.concat([t1_conv2, t1_conv19], axis=3)
            t1_conv20 = gen_conv(t1_conv19, cnum//2, 3, 1, name='t1conv20')
            
            x_score1 = gen_conv(t1_conv20, 1, 3, 1, name='score1')
            

            # stage1
            xnow = tf.concat([xin, ones_x, x_score1], axis=3)
            s1_conv1 = gen_conv(xnow, cnum, 5, 1, name='conv1')
            s1_conv2 = gen_conv(s1_conv1, 2*cnum, 3, 2, name='conv2_downsample')
            s1_conv3 = gen_conv(s1_conv2, 2*cnum, 3, 1, name='conv3')
            s1_conv4 = gen_conv(s1_conv3, 4*cnum, 3, 2, name='conv4_downsample')
            s1_conv5 = gen_conv(s1_conv4, 4*cnum, 3, 1, name='conv5')
            s1_conv6 = gen_conv(s1_conv5, 4*cnum, 3, 1, name='conv6')
            # mask_s = resize_mask_like(mask, s1_conv6)
            s1_conv7 = res_block(s1_conv6, name='s1res_block1')
            s1_conv8 = res_block(s1_conv7, name='s1res_block2')
            s1_conv9 = res_block(s1_conv8, name='s1res_block3')
            s1_conv10 = res_block(s1_conv9, name='s1res_block4')
            s1_conv11 = gen_conv(s1_conv10, 4*cnum, 3, 1, name='conv11')
            s1_conv11 = tf.concat([s1_conv6, s1_conv11], axis=3)
            s1_conv12 = gen_conv(s1_conv11, 4*cnum, 3, 1, name='conv12')
            s1_conv12 = tf.concat([s1_conv5, s1_conv12], axis=3)
            s1_conv13 = gen_deconv(s1_conv12, 2*cnum, name='conv13_upsample')
            s1_conv13 = tf.concat([s1_conv3, s1_conv13], axis=3)
            s1_conv14 = gen_conv(s1_conv13, 2*cnum, 3, 1, name='conv14')
            s1_conv14 = tf.concat([s1_conv2, s1_conv14], axis=3)
            s1_conv15 = gen_deconv(s1_conv14, cnum, name='conv15_upsample')
            s1_conv15 = tf.concat([s1_conv1, s1_conv15], axis=3)
            s1_conv16 = gen_conv(s1_conv15, cnum//2, 3, 1, name='conv16')
            s1_conv17 = gen_conv(s1_conv16, 3, 3, 1, activation=None, name='conv17')
            s1_conv = tf.clip_by_value(s1_conv17, -1., 1., name='stage1')
            #s1_conv = tf.nn.tanh(s1_conv17, name='stage1')
            x_stage1 = s1_conv

        return x_stage1, x_score1

    def build_sngan_discriminator(self, x, reuse=False, training=True):
        with tf.variable_scope('discriminator', reuse=reuse):
            cnum = 64
            x = sndis_conv(x, cnum, 5, 1, name='conv1', training=training)
            x = sndis_conv(x, cnum*2, 5, 2, name='conv2', training=training)
            x = sndis_conv(x, cnum*4, 5, 2, name='conv3', training=training)
            x = sndis_conv(x, cnum*4, 5, 2, name='conv4', training=training)
            x = sndis_conv(x, cnum*4, 5, 2, name='conv5', training=training)
            x = sndis_conv(x, cnum*4, 5, 2, name='conv6', training=training)
            return x