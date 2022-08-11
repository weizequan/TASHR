import os
import random
import numpy as np
import tensorflow as tf
from lite_metrics import Progbar
from config import Config
from data_load import Dataset
from lite_model import Removal



def main():
    config_path = os.path.join('config.yml')
    config = Config(config_path)
    config.print()
    # Init cuda environment
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    # Init random seed to less result random
    tf.set_random_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # Init training data

    dataset = Dataset(config)
    batch_concat = dataset.batch_concat
    val_concat = dataset.val_concat

    # Init the model
    model = Removal(config)
    gen_loss,dis_loss, t_psnr, t_ssim = model.build_whole_model(batch_concat)
    gen_optim, dis_optim = model.build_optim(gen_loss, dis_loss)
    val_psnr,val_ssim = model.build_validation_model(val_concat)

    # Create the graph
    config_graph = tf.ConfigProto()
    config_graph.gpu_options.allow_growth = True

    with tf.Session(config=config_graph) as sess:
        # Merge all the summaries
        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter(config.CHECKPOINTS + 'train', sess.graph)
        eval_writer = tf.summary.FileWriter(config.CHECKPOINTS + 'eval')
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        checkpoint = tf.train.get_checkpoint_state(config.LOAD_MODEL)

        if (checkpoint and checkpoint.model_checkpoint_path):
            print(checkpoint.model_checkpoint_path)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(config.LOAD_MODEL))
            epoch = int(meta_graph_path.split("-")[-1].split(".")[0])
            step = int(epoch * dataset.len_train / dataset.batch_size)
        else:
            step = 0
            epoch = 0

        # load detection network weights
        checkpoints_path = './ocr/ctpn/checkpoints'
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        num = 0
        for var in vars_list:
            vname = var.name
            from_name = vname
            if vname.find('text_detector') != -1:
                from_name = vname.replace('text_detector/','')
                var_value = tf.contrib.framework.load_variable(checkpoints_path, from_name)
                assign_ops.append(tf.assign(var, var_value))
                num += 1
        sess.run(assign_ops)
        print(num)
        print('Model loaded.')

        
        # Start input enqueue threads
        progbar = Progbar(dataset.len_train, width=20, stateful_metrics=['epoch', 'iter', 'gen_loss', 'dis_loss', 'psnr', 'ssim'])
        while epoch < config.EPOCH:
            step += 1
            epoch = int(step / dataset.len_train * dataset.batch_size)

            g_loss, _ = sess.run([gen_loss, gen_optim])
            d_loss, _ = sess.run([dis_loss, dis_optim])
            tr_psnr, tr_ssim = sess.run([t_psnr, t_ssim])
            logs = [
                ("epoch", epoch),
                ("iter", step),
                ("g_loss", g_loss),
                ("d_loss", d_loss),
                ("psnr", tr_psnr),
                ("ssim", tr_ssim)
            ]
            progbar.add(dataset.batch_size, values=logs)
            if step % (dataset.len_train / dataset.batch_size) == 0:
                progbar = Progbar(dataset.len_train, width=20, stateful_metrics=['epoch', 'iter', 'gen_loss', 'dis_loss', 'psnr', 'ssim'])
            if step % config.SUMMARY_INTERVAL == 0:
                # Run validation
                v_psnr = []
                v_ssim = []
                for i in range(dataset.len_val // dataset.val_batch_size):
                    val_psnr_tmp,val_ssim_tmp=sess.run([val_psnr,val_ssim])
                    v_psnr.append(val_psnr_tmp)
                    v_ssim.append(val_ssim_tmp)
                eval_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='val_psnr', simple_value=np.mean(v_psnr))]), epoch)
                eval_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='val_ssim', simple_value=np.mean(v_ssim))]), epoch)

                # Train summary
                summary = sess.run(merged)           # , options=run_options, run_metadata=run_metadata)
                train_writer.add_summary(summary, epoch)
            if step % config.SAVE_INTERVAL == 0:
                if (checkpoint and checkpoint.model_checkpoint_path):
                    saver.save(sess, config.CHECKPOINTS + 'removal', global_step=epoch, write_meta_graph=False)
                else:
                    saver.save(sess, config.CHECKPOINTS + 'removal', global_step=epoch, write_meta_graph=True)



    sess.close()


if __name__ == "__main__":
    main()
