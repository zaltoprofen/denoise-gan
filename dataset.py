from chainer.dataset import DatasetMixin
from PIL import Image
import numpy as np
import csv


class TupleImageDataset(DatasetMixin):
    def __init__(self, csv_path, normalize=True, dtype=np.float32):
        with open(csv_path) as fp:
            r = csv.reader(fp)
            self._imgs_paths = list(r)

        self.dtype = dtype
        self.normalize = normalize

    def _read(self, path):
        img = Image.open(path)  # type: Image.Image
        img_ = img.convert('RGB')
        data = np.asarray(img_, dtype=self.dtype).transpose(2, 0, 1)
        if self.normalize:
            data = data / 127.5 - 1.0
        img_.close()
        img.close()
        return data

    def __len__(self):
        return len(self._imgs_paths)

    def get_example(self, i):
        return tuple(self._read(img) for img in self._imgs_paths[i])
