import gan
import chainer
import chainer.functions as F
import glob
import numpy as np

import my_extension
from dataset import TupleImageDataset
from PIL import Image
from chainer.training import extensions


class CGANLossFunctions(gan.GANLossFunctions):
    def __init__(self, lambda_=100.0):
        self.lambda_ = lambda_

    def generator_loss(self, generated, *args, **kwargs):
        x1 = args[0]  # type: chainer.Variable
        x2 = args[1]  # type: chainer.Variable
        xp = chainer.cuda.get_array_module(generated.data)
        batch_size = generated.shape[0]
        return F.sigmoid_cross_entropy(
            self.dis(x1, generated), xp.ones((batch_size, 1, 1, 1), dtype=xp.int32)) \
               + self.lambda_ * F.mean_absolute_error(generated, x2)

    def discriminator_loss(self, generated, *args, **kwargs):
        x1 = args[0]  # type: chainer.Variable
        x2 = args[1]  # type: chainer.Variable
        xp = chainer.cuda.get_array_module(generated.data)
        batch_size = x1.shape[0]
        return F.sigmoid_cross_entropy(
            self.dis(x1, generated), xp.zeros((batch_size, 1, 1, 1), dtype=xp.int32)) \
               + F.sigmoid_cross_entropy(self.dis(x1, x2),
                                         xp.ones((batch_size, 1, 1, 1), dtype=xp.int32))


def to_ndarray(path):
    img = Image.open(path)  # type: Image.Image
    img_ = img.convert('RGB')
    a = np.asarray(img_, dtype=np.float32).transpose(2, 0, 1)
    img.close()
    img_.close()
    return a / 127.5 - 1.0


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--use-gpu', action='store_true')
    args = p.parse_args()

    gen = gan.Translator()
    dis = gan.ConditionalDiscriminator()

    if args.use_gpu:
        gen.to_gpu(0)
        dis.to_gpu(0)

    gopt = chainer.optimizers.Adam()
    gopt.setup(gen)

    dopt = chainer.optimizers.Adam()
    dopt.setup(dis)

    dataset = TupleImageDataset('dataset.csv')
    iterator = chainer.iterators.MultiprocessIterator(dataset, 50, n_processes=10, n_prefetch=3)

    updater = gan.GANUpdater(
        iterator,
        gen,
        gopt,
        dis,
        dopt,
        100.0,
        device=0 if args.use_gpu else None,
    )
    t = chainer.training.Trainer(updater, (10, 'epoch'))
    t.extend(extensions.dump_graph('gen/loss', 'gen.dot'))
    t.extend(extensions.dump_graph('dis/loss', 'dis.dot'))
    t.extend(extensions.LogReport(trigger=(1, 'epoch')))
    t.extend(extensions.PlotReport(['gen/loss', 'dis/loss']))
    t.extend(my_extension.eval_image(gen, dataset, device=0), trigger=(1, 'epoch'))
    t.extend(extensions.snapshot(trigger=(10, 'epoch')))
    t.run()

    chainer.serializers.save_npz('gen.npz', gen)
    chainer.serializers.save_npz('dis.npz', dis)


def to_npy(glob_query, npy_filename):
    y = np.concatenate([to_ndarray(f) for f in glob.glob(glob_query)])
    np.save(npy_filename, y)


if __name__ == '__main__':
    # to_npy('Illustration_256/*.png', '256.npy')
    # to_npy('Illustration_512/*.png', '512.npy')
    main()
