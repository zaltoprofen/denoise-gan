import chainer
from chainer.training import extensions

import gan
import my_extension
from dataset import TupleImageDataset


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


if __name__ == '__main__':
    # to_npy('Illustration_256/*.png', '256.npy')
    # to_npy('Illustration_512/*.png', '512.npy')
    main()
