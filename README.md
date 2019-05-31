# 验证码识别
使用 TensorFlow 深度学习破解验证码

## 训练
运行命令开始训练，不使用 GPU 需要 20 多个小时，文末附训练结果下载。 

```
python ./train_captcha.py [resume]
```
PS: resume: true 加载上次中断训练的模型继续训练

## 测试
使用 `python .\generate_captcha.py` 会在当前目录下生成一张测试的验证码图片
执行以下命令获取识别结果
```
python predict_captcha.py 图片名称.jpg
```

## 训练结果下载
链接：[https://pan.baidu.com/s/1GmE3tq5m-psqH5o-yPK3sA](https://pan.baidu.com/s/1GmE3tq5m-psqH5o-yPK3sA)

提取码：`aq35`

将下载的压缩包解压放到根目录，得到 `model` 文件夹。
