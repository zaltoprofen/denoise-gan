import sys
from PIL import Image
import os


def rip(in_file, out_file_prefix, x_sample=3):
    img = Image.open(in_file)  # type: Image.Image
    w, h = img.size
    wc, hc = int(w / 2), int(h / 2)
    ws, hs = int((w - wc) / (x_sample-1)), int((h-hc) / (x_sample-1))

    for i in range(x_sample):
        bb = [ws*i, 0, ws*i+wc, 0]
        for j in range(x_sample):
            bb[1], bb[3] = hs*j, hs*j+hc
            img_ = img.crop(bb)
            img_.save(out_file_prefix + '_{}_{}.png'.format(i, j))
            img_.close()
    img.close()


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('input_dir')
    p.add_argument('output_dir')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--x-sample', default=3, type=int)
    args = p.parse_args()

    if not args.overwrite:
        if os.path.exists(args.output_dir):
            print('directory {} is already existed'.format(args.output_dir), file=sys.stderr)
            sys.exit(1)
        os.makedirs(args.output_dir)

    for dirpath, _, files in os.walk(args.input_dir):
        for img in files:
            e = os.path.splitext(img)
            if e[1] not in ['.jpg', '.png', '.jpeg']:
                continue
            rip(os.path.join(dirpath, img), os.path.join(args.output_dir, e[0]), args.x_sample)


if __name__ == '__main__':
    main()
