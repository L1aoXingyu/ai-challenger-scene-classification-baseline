import datetime
import json
import os

import mxnet as mx
import numpy as np
from mxnet import gluon as gl
from mxnet import nd

from data_utils import SceneDataSet
from tensorboardX import SummaryWriter

train_root = './data/ai_challenger_scene_train_20170904/'
valid_root = './data/ai_challenger_scene_validation_20170908/'
test_root = './data/ai_challenger_scene_test_a_20170922/'


def transform_train(img):
    '''
    img is the mx.image.imread object
    '''
    img = img.astype('float32') / 255
    random_shape = int(np.random.uniform() * 224 + 256)
    # random samplely in [256, 480]
    aug_list = mx.image.CreateAugmenter(
        data_shape=(3, 224, 224),
        resize=random_shape,
        rand_mirror=True,
        rand_crop=True,
        mean=np.array([0.4960, 0.4781, 0.4477]),
        std=np.array([0.2915, 0.2864, 0.2981]))

    for aug in aug_list:
        img = aug(img)
    img = nd.transpose(img, (2, 0, 1))
    return img


def transform_valid(img):
    img = img.astype('float32') / 255.
    aug_list = mx.image.CreateAugmenter(
        data_shape=(3, 224, 224),
        mean=np.array([0.4960, 0.4781, 0.4477]),
        std=np.array([0.2915, 0.2864, 0.2981]))

    for aug in aug_list:
        img = aug(img)
    img = nd.transpose(img, (2, 0, 1))
    return img


# ## use DataLoader

train_json = train_root + '/scene_train_annotations_20170904.json'
train_img_path = train_root + '/scene_train_images_20170904/'
train_set = SceneDataSet(train_json, train_img_path, transform_train)
train_data = gl.data.DataLoader(
    train_set, batch_size=64, shuffle=True, last_batch='keep')

valid_json = valid_root + '/scene_validation_annotations_20170908.json'
valid_img_path = valid_root + '/scene_validation_images_20170908/'
valid_set = SceneDataSet(valid_json, valid_img_path, transform_valid)
valid_data = gl.data.DataLoader(
    valid_set, batch_size=128, shuffle=False, last_batch='keep')

# train_img_path = train_root + '/scene_train_images_20170904/'
# train_iter = mx.image.ImageIter(
#     batch_size=64,
#     data_shape=(3, 224, 224),
#     path_imglist='./train_list.lst',
#     path_root=train_img_path,
#     shuffle=True)
# train_iter.augmentation_transform = transform_train

# valid_img_path = valid_root + '/scene_validation_images_20170908/'
# valid_iter = mx.image.ImageIter(
#     batch_size=64,
#     data_shape=(3, 224, 224),
#     path_imglist='./valid_list.lst',
#     path_root=valid_img_path,
#     shuffle=False)
# valid_iter.augmentation_transform = transform_valid

criterion = gl.loss.SoftmaxCrossEntropyLoss()

# ctx = [mx.gpu(0), mx.gpu(1)]
ctx = mx.gpu(0)
num_epochs = 100
lr = 0.1
wd = 1e-4
lr_decay = 0.1

net = gl.model_zoo.vision.resnet50_v2(classes=80)
net.initialize(init=mx.init.Xavier(), ctx=ctx)
net.hybridize()

writer = SummaryWriter()


def get_acc(output, label):
    pred = output.argmax(1)
    correct = (pred == label).sum()
    return correct.asscalar()


def train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_decay):
    trainer = gl.Trainer(net.collect_params(), 'sgd',
                         {'learning_rate': lr,
                          'momentum': 0.9,
                          'wd': wd})

    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        # train_data.reset()
        # valid_data.reset()
        train_loss = 0
        correct = 0
        total = 0
        for data, label in train_data:
            # for batch in train_data:
            # data = batch.data[0].as_in_context(ctx)
            # label = batch.label[0].as_in_context(ctx)
            bs = data.shape[0]
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # data_list = gl.utils.split_and_load(data, ctx, even_split=False)
            # label_list = gl.utils.split_and_load(label, ctx, even_split=False)
            with mx.autograd.record():
                output = net(data)
                loss = criterion(output, label)

            # outputs = [net(X) for X in data_list]
            # losses = [criterion(output, y) for output, y in zip(outputs, label_list)]
            # for l in losses:
            #     l.backward()
            loss.backward()
            trainer.step(bs)
            train_loss += loss.sum().asscalar()
            # train_loss += sum([l.sum().asscalar() for l in losses]) / bs
            # correct += sum([
            #     get_acc(output.as_in_context(ctx[0]), y.as_in_context(ctx[0]))
            #     for output, y in zip(outputs, label_list)
            # ])
            correct += get_acc(output, label)
            total += bs
        writer.add_scalars('loss', {'train': train_loss / total}, epoch)
        writer.add_scalars('acc', {'train': correct / total}, epoch)
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_correct = 0
            valid_total = 0
            valid_loss = 0
            for data, label in valid_data:
                # for batch in valid_data:
                # data = batch.data[0].as_in_context(ctx)
                # label = batch.label[0].as_in_context(ctx)
                bs = data.shape[0]
                data = data.as_in_context(ctx)
                label = label.as_in_context(ctx)
                output = net(data)
                loss = criterion(output, label)
                valid_loss += nd.sum(loss).asscalar()
                valid_correct += get_acc(output, label)
                valid_total += bs
            valid_acc = valid_correct / valid_total
            writer.add_scalars('loss', {'valid': valid_loss / total}, epoch)
            writer.add_scalars('acc', {'valid': valid_acc}, epoch)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train acc %f, Valid Loss: %f, Valid acc %f, "
                % (epoch, train_loss / total, correct / total,
                   valid_loss / valid_total, valid_acc))
        else:
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, " %
                         (epoch, train_loss / total, correct / total))
        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))


train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_decay)

net.save_params('./net.params')