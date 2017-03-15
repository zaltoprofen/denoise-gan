from typing import Union, Tuple, Callable, Optional, Any

import chainer
import chainer.links as L
import chainer.functions as F

if chainer.cuda.available:
    import cupy

import numpy

from chainer.dataset.iterator import Iterator
from chainer import reporter


class BNConvolution2D(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, wscale=1, bias=0, nobias=False,
                 use_cudnn=True, initialW=None, initial_bias=None, deterministic=False):
        super().__init__(
            conv=L.Convolution2D(in_channels, out_channels, ksize, stride, pad, wscale, bias, nobias, use_cudnn,
                                 initialW, initial_bias, deterministic),
            bn=L.BatchNormalization(out_channels),
        )

    def __call__(self, x, test=False):
        return self.bn(self.conv(x), test=test)


class BNDeconvolution2D(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, wscale=1, bias=0, nobias=False, outsize=None,
                 use_cudnn=True, initialW=None, initial_bias=None, deterministic=False):
        super().__init__(
            deconv=L.Deconvolution2D(in_channels, out_channels, ksize, stride, pad, wscale, bias, nobias, outsize,
                                     use_cudnn, initialW, initial_bias, deterministic),
            bn=L.BatchNormalization(out_channels),
        )

    def __call__(self, x, test=False):
        return self.bn(self.deconv(x), test=test)

class BNDConvolution2D(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, wscale=1, bias=0, nobias=False,
                 dropout_ratio=0.5, always_dropout=False, use_cudnn=True, initialW=None, initial_bias=None,
                 deterministic=False):
        super().__init__(
            conv=L.Convolution2D(in_channels, out_channels, ksize, stride, pad, wscale, bias, nobias, use_cudnn,
                                 initialW, initial_bias, deterministic),
            bn=L.BatchNormalization(out_channels),
        )
        self._drop_ratio = dropout_ratio
        self._always_dropout = always_dropout

    def __call__(self, x, test=False):
        d_train = self._always_dropout or not test
        return F.dropout(self.bn(self.conv(x), test=test), self._drop_ratio, d_train)


class BNDDeconvolution2D(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, wscale=1, bias=0, nobias=False, outsize=None,
                 dropout_ratio=0.5, always_dropout=False, use_cudnn=True, initialW=None, initial_bias=None,
                 deterministic=False):
        super().__init__(
            deconv=L.Deconvolution2D(in_channels, out_channels, ksize, stride, pad, wscale, bias, nobias, outsize,
                                     use_cudnn, initialW, initial_bias, deterministic),
            bn=L.BatchNormalization(out_channels),
        )
        self._drop_ratio = dropout_ratio
        self._always_dropout = always_dropout

    def __call__(self, x, test=False):
        d_train = self._always_dropout or not test
        return F.dropout(self.bn(self.deconv(x), test=test), self._drop_ratio, d_train)


class ConditionalDiscriminator(chainer.Chain):
    def __init__(self):
        super().__init__(
            conv1_1=BNConvolution2D(3, 32, 4, 2, 1),  # 256
            conv1_2=BNConvolution2D(3, 32, 4, 2, 1),
            conv2=BNConvolution2D(64, 128, 4, 2, 1),  # 128
            conv3=BNConvolution2D(128, 256, 4, 2, 1),  # 64

            determine=L.Convolution2D(256, 1, 5),

            # fc=L.Linear(32 * 32 * 2048, 1),
        )

    def __call__(self, x1: chainer.Variable, x2: chainer.Variable):
        assert x1.shape == x2.shape

        h1 = F.concat([
            F.leaky_relu(self.conv1_1(x1)),
            F.leaky_relu(self.conv1_2(x2)),
        ])
        h2 = F.leaky_relu(self.conv2(h1))
        h3 = F.leaky_relu(self.conv3(h2))

        det = self.determine(h3)

        # return F.average_pooling_2d(det, det.shape[2:])
        return det


class Translator(chainer.Chain):
    def __init__(self):
        super().__init__(
            conv_in=L.Convolution2D(3, 32, 1),

            conv1=BNConvolution2D(32, 32, 4, 2, 1),  # 128 -> 64
            conv2=BNConvolution2D(32, 64, 4, 2, 1),  # 64 -> 32
            conv3=BNConvolution2D(64, 128, 4, 2, 1),  # 32 -> 16

            deconv3=BNDDeconvolution2D(128, 64, 4, 2, 1, always_dropout=True),  # 16 -> 32
            deconv2=BNDDeconvolution2D(64 + 64, 32, 4, 2, 1, always_dropout=True),  # 32 -> 64
            deconv1=BNDeconvolution2D(32 + 32, 32, 4, 2, 1),  # 64 -> 128

            deconv_out=L.Deconvolution2D(32, 3, 1),
        )

    def __call__(self, x, test=False):
        x = self.conv_in(x)

        e1 = F.leaky_relu(self.conv1(x, test=test))
        e2 = F.leaky_relu(self.conv2(e1, test=test))
        e3 = F.leaky_relu(self.conv3(e2, test=test))

        d3 = F.leaky_relu(self.deconv3(e3, test=test))
        d2 = F.leaky_relu(self.deconv2(F.concat([d3, e2[:, :, :d3.shape[2], :d3.shape[3]]]), test=test))
        d1 = F.leaky_relu(self.deconv1(F.concat([d2, e1[:, :, :d2.shape[2], :d2.shape[3]]]), test=test))

        return F.clip(self.deconv_out(d1), -1.0, 1.0)


class GANLossFunctions:
    def __init__(self):
        pass

    def generator_loss(self, generated: chainer.Variable, *args, **kwargs):
        xp = chainer.cuda.get_array_module(generated.data)
        batch_size = generated.shape[0]
        return F.sigmoid_cross_entropy(
            self.dis(generated), xp.ones((batch_size, 1, 1, 1), dtype=xp.int32))

    def discriminator_loss(self, generated: chainer.Variable, *args, **kwargs):
        batch = args[0]  # type: chainer.Variable
        xp = chainer.cuda.get_array_module(batch.data)
        batch_size = batch.shape[0]
        return F.sigmoid_cross_entropy(
            self.dis(generated), xp.zeros((batch_size, 1, 1, 1), dtype=xp.int32)) \
               + F.sigmoid_cross_entropy(self.dis(batch),
                                         xp.ones((batch_size, 1, 1, 1), dtype=xp.int32))


class GANUpdater(chainer.training.StandardUpdater):
    def __init__(self, iterator: Iterator,
                 gen: Union[chainer.Chain, Callable[..., chainer.Variable]],
                 gen_optimizer: chainer.Optimizer,
                 dis: Union[chainer.Chain, Callable[..., chainer.Variable]],
                 dis_optimizer: chainer.Optimizer,
                 lambda_: float = 100.0,
                 device: Optional[int] = None):
        self.gen = gen
        self.dis = dis
        optimizers = {
            'gen': gen_optimizer,
            'dis': dis_optimizer,
        }
        self.lambda_ = lambda_
        super().__init__(iterator, optimizers, device=device)

    def update_core(self):
        iterator = self.get_iterator('main')  # type: Iterator
        g_opt = self.get_optimizer('gen')  # type: chainer.Optimizer
        d_opt = self.get_optimizer('dis')  # type: chainer.Optimizer

        batch = iterator.next()
        batch_size = len(batch)
        x1, x2 = self.converter(batch, self.device)

        generated = self.gen(x1)
        dis_real = self.dis(x1, x2)
        dis_fake = self.dis(x1, generated)

        g_loss = F.sum(F.softplus(-dis_fake))/dis_fake.size + self.lambda_ * F.mean_absolute_error(generated, x2)
        reporter.report({'loss': g_loss}, self.gen)
        self.gen.cleargrads()
        g_loss.backward()
        g_opt.update()
        del g_loss

        d_loss = F.sum(F.softplus(-dis_real))/dis_real.size + F.sum(F.softplus(dis_fake))/dis_fake.size
        reporter.report({'loss': d_loss}, self.dis)
        self.dis.cleargrads()
        d_loss.backward()
        d_opt.update()
        del d_loss, dis_fake, dis_real, x1, x2, batch, generated
