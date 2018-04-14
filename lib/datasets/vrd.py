from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
import os.path as osp
import numpy as np
import scipy.misc as misc
import scipy.sparse
import scipy.io as sio
# import utils.cython_bbox
import cPickle
import subprocess

from .imdb import imdb
from . import ds_utils
from model.utils.config import cfg

# from datasets.imdb import imdb
# import datasets.ds_utils as ds_utils
# from model.config import cfg

class vrd(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, 'vrd_' + image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path()
        self._data_path = osp.join(self._devkit_path, 'images/%s' % image_set)
        self._classes = ['__background__'] + self._object_classes()
        self._class_to_ind = dict(zip(self._classes, xrange(self.num_classes)))
        
        with open('data/VRD/%s.pkl'%(image_set), 'rb') as fid:
            anno = cPickle.load(fid)
            self._anno = [x for x in anno if x is not None]

        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def _object_classes(self):
        classes = [x.strip() for x in open(osp.join(self._devkit_path, 'obj.txt')).readlines()]
        return classes

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_index[i] 

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_index = []
        for img_idx, anno_img in enumerate(self._anno):
          image_path = anno_img['img_path'].split('/')[-1]
          image_path = osp.join('data/VRD/images/', self._image_set, image_path)
          assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
          image_index.append(image_path)
        return image_index

    def _get_default_path(self):
        """
        Return the default path where VRD is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VRD')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self._load_vrd_annotation()

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)
    
    def _load_vrd_annotation(self):
        """
        Load image and bounding boxes info from VRD format.
        """
        gt_roidb = []
        for img_idx, anno_img in enumerate(self._anno):
            img_height, img_width = misc.imread(self.image_path_at(img_idx)).shape[0:2]
            num_objs = anno_img['boxes'].shape[0]
            boxes = anno_img['boxes']
            gt_classes = anno_img['classes'] + 1 # add background category
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            # "Seg" area for pascal is just the box area
            seg_areas = np.zeros((num_objs), dtype=np.float32)

            for ix in range(num_objs):
                x1,y1,x2,y2 = boxes[ix]
                overlaps[ix, gt_classes[ix]] = 1.0
                seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

            overlaps = scipy.sparse.csr_matrix(overlaps)

            gt_roidb.append({'boxes' : boxes,                             
                             'gt_classes': gt_classes,
                             'gt_overlaps' : overlaps,
                             'flipped' : False,
                             'seg_areas' : seg_areas})
        return gt_roidb

    def evaluate_detections(self, all_boxes, output_dir):
        det_file = osp.join(output_dir, 'proposal_faster_rcnn.pkl')
        # if os.path.exists(det_file):
        #     with open(det_file, 'rb') as fid:
        #         proposals = cPickle.load(fid)
        #     print '{} ss roidb loaded from {}'.format(self.name, det_file)            
        # else:
        proposals = {}
        proposals['boxes'] = []
        proposals['confs'] = []
        proposals['cls'] = []
        for ii in range(self.num_images):
            box = np.zeros((0,4))
            cls = []
            confs = np.zeros((0,1))
            for jj in range(1, self.num_classes):
                box = np.vstack((box, all_boxes[jj][ii][:, 0:4]))
                confs = np.vstack((confs, all_boxes[jj][ii][:, 4:5]))
                for kk in range(all_boxes[jj][ii].shape[0]):
                    cls.append(jj-1)
            proposals['boxes'].append(box)
            proposals['confs'].append(confs)
            proposals['cls'].append(cls)
        with open(det_file, 'wb') as f:
            cPickle.dump(proposals, f, cPickle.HIGHEST_PROTOCOL)

        res = self.evaluate_recall(candidate_boxes=proposals['boxes'], thresholds=[0.5], area='all')
        out_name   = output_dir.split('/')[-1]
        output_dir = '/'.join(output_dir.split('/')[0:-1])
        with open(osp.join(output_dir, 'recall.txt'), 'a') as f:
            f.write('%s, Recall:%f\n'%(out_name, res['recalls'][0]))

if __name__ == '__main__':
    imdb = vrd_det('test')
    from IPython import embed; embed()
    #proposal_method = 'gt'
    #if(proposal_method == 'edge_box'):
    #    proposals = sio.loadmat('proposals_edge_box.mat')['proposals'][0]
    #else:
    #    with open('/home/data/liangkongming/sg_dataset/proposals/proposal_%s.pkl'%proposal_method, 'rb') as fid:
    #        proposals = cPickle.load(fid)
    #        if(proposal_method == 'eccv'):
    #            proposals = proposals['boxes'][0]
    #        else:
    #            proposals = proposals['boxes']
    ##for limit in range(50,100,50):
    #res = imdb.evaluate_recall(candidate_boxes=proposals, thresholds=[0.5], area='all')
    #print res['recalls'][0]
    #imdb = vrd_det('test', '/home/data/liangkongming/sg_dataset')

    # with open('/home/code/liangkongming/caffe/output/faster_rcnn_end2end/vrd_test/vgg16_faster_rcnn_iter_70000/proposals.pkl', 'rb') as fid:
    #     proposals = cPickle.load(fid)
    # imdb = vrd_det('test')
    # res = imdb.evaluate_recall(candidate_boxes=proposals, thresholds=[0.5], area='all')
    # print res['recalls'][0]
    # from IPython import embed; embed()
