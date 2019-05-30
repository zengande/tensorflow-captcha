import tensorflow as tf
import numpy as np
import string
import generate_captcha
import captcha_model
import sys

ACCURACY = 0.99


def __isResume():
    if len(sys.argv) > 1:
        return sys.argv[1] == 'true'
    else:
        return False


if __name__ == '__main__':

    resume = __isResume()
    captcha = generate_captcha.generateCaptcha()
    width, height, char_num, characters, classes = captcha.get_parameter()

    x = tf.placeholder(tf.float32, [None, height, width, 1])
    y_ = tf.placeholder(tf.float32, [None, char_num*classes])
    keep_prob = tf.placeholder(tf.float32)

    model = captcha_model.captchaModel(width, height, char_num, classes)
    y_conv = model.create_model(x, keep_prob)
    cross_entropy = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    predict = tf.reshape(y_conv, [-1, char_num, classes])
    real = tf.reshape(y_, [-1, char_num, classes])
    correct_prediction = tf.equal(tf.argmax(predict, 2), tf.argmax(real, 2))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if resume:
            saver.restore(sess, "model/capcha_model.ckpt")
        merged = [tf.summary.merge_all()]  # 将图形、训练过程等数据合并在一起
        train_writer = tf.summary.FileWriter(
            'logs', sess.graph)  # 将训练日志写入到logs文件夹下

        step = 0
        while True:
            batch_x, batch_y = next(captcha.gen_captcha(64))
            _, loss = sess.run([train_step, cross_entropy], feed_dict={
                               x: batch_x, y_: batch_y, keep_prob: 0.75})
            print('step:%d,loss:%f' % (step, loss))
            if step % 100 == 0:
                batch_x_test, batch_y_test = next(captcha.gen_captcha(100))
                acc = sess.run(accuracy, feed_dict={
                               x: batch_x_test, y_: batch_y_test, keep_prob: 1.})
                print('========= step:%d,accuracy:%f =========' % (step, acc))
                if step > 0:
                    saver.save(sess, "model/capcha_model.ckpt")
                if acc > ACCURACY:
                    break
            step += 1
