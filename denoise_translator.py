import chainer
import chainer.functions as F
from chainer.training import extensions
from chainer import reporter

import gan
import my_extension
from dataset import TupleImageDataset


def loss(translator: gan.Translator):
    def _loss(x1, x2):
        translated = translator(x1)
        val = F.mean_absolute_error(translated, x2)
        reporter.report({'loss': val}, translator)
        return val
    return _loss


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--use-gpu', action='store_true')
    args = p.parse_args()

    gen = gan.Translator()

    if args.use_gpu:
        gen.to_gpu(0)

    gopt = chainer.optimizers.Adam()
    gopt.setup(gen)

    dataset = TupleImageDataset('dataset.csv')
    iterator = chainer.iterators.MultiprocessIterator(dataset, 50, n_processes=10, n_prefetch=3)

    updater = chainer.training.StandardUpdater(iterator, gopt, device=0 if args.use_gpu else None, loss_func=loss(gen))
    t = chainer.training.Trainer(updater, (10, 'epoch'), out='only_translator')
    t.extend(extensions.dump_graph('main/loss', 'translator.dot'))
    t.extend(extensions.LogReport(trigger=(1, 'epoch')))
    t.extend(extensions.PlotReport(['main/loss']))
    t.extend(my_extension.eval_image(gen, dataset, device=0), trigger=(1, 'epoch'))
    t.extend(extensions.snapshot(trigger=(10, 'epoch')))
    t.run()

    chainer.serializers.save_npz('only_translator.npz', gen)


if __name__ == '__main__':
    # to_npy('Illustration_256/*.png', '256.npy')
    # to_npy('Illustration_512/*.png', '512.npy')
    main()
