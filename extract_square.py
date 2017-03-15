import sys
from PIL import Image
import os


def extract_square(in_file, out_file, size=(512, 512)):
    assert in_file != out_file
    img = Image.open(in_file)  # type: Image.Image
    aspect_ratio = size[0] / size[1]
    w, h = img.size
    if w < size[0] or h < size[1]:
        print('skip:', in_file, file=sys.stderr)
        return

    wc, hc = w / 2, h / 2
    if aspect_ratio > w / h:
        height = w / aspect_ratio
        bb = (0, hc - height / 2, w, hc + height / 2)
    else:
        width = h * aspect_ratio
        bb = (wc - width / 2, 0, wc + width / 2, h)
    img = img.crop(bb)
    img = img.resize(size, Image.BICUBIC)
    img.save(out_file)
    img.close()


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('input_dir')
    p.add_argument('output_dir')
    p.add_argument('--overwrite', action='store_true')
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
            extract_square(os.path.join(dirpath, img), os.path.join(args.output_dir, e[0] + '.png'))


if __name__ == '__main__':
    main()
