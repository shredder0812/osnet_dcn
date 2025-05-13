from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings

from ..dataset import ImageDataset
import torchreid

class EndoCV(ImageDataset):
    """EndoCV.

    Reference:
        Cao Thanh Tung
    Dataset statistics:
        - identities: 51
        - images: 830 (train) + 54 (query) + 223 (gallery).
    """
    #_junk_pids = [0, -1]
    dataset_dir = 'endocv'
    dataset_url = 'https://drive.google.com/file/d/13I-Juawl9RTbN3TN1E4vlPLa_EwvHTtE/view?usp=sharing'

    def __init__(self, root='', endocv_distractor=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'EndoCV')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "bounding_box_train" under '
                '"Market-1501-v15.09.15".'
            )

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.endocv_distractor = endocv_distractor

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        if self.endocv_distractor:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        gallery = [(x[0], x[1], 1) for x in gallery]  # camera view 1
        if self.endocv_distractor:
            gallery += self.process_dir(self.extra_gallery_dir, relabel=False)

        super(EndoCV, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            assert 0 <= pid <= 1000 # pid == 0 means background
            assert 1 <= camid <= 10
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data
