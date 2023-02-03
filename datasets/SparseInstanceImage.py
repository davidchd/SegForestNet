import numpy as np
import rust


class SparseInstanceImage():
    def __init__(self, gt_rgb, inst_rgb, class_map):
        assert class_map.shape[0] <= 256
        
        gt_rgb = np.asarray(gt_rgb)
        inst_rgb = np.asarray(inst_rgb)
        
        gt = np.empty(gt_rgb.shape[:2], dtype=np.uint8)
        inst = np.empty(gt.shape, dtype=np.int32)
        num_inst = rust.rgb2ids(gt, gt_rgb, inst, inst_rgb, class_map)
                
        self.instances = np.empty((num_inst, 6), dtype=np.uint64)
        rows = np.zeros((gt.shape[0], num_inst+1), dtype=np.int32)
        cols = np.zeros((gt.shape[1], num_inst+1), dtype=np.int32)
        rust.extract_instances(self.instances, rows, cols, gt, inst)
        
        self.rows = rows[:,1:np.max(rows[:,0])+1].copy()
        self.cols = cols[:,1:np.max(cols[:,0])+1].copy()
        assert self.rows.flags['C_CONTIGUOUS'] and self.cols.flags['C_CONTIGUOUS']

        masks = [np.where(inst[y0:y1,x0:x1]==i+1, 1, 0) for i, (c, y0, y1, x0, x1, offset) in enumerate(self.instances)]
        masks = [np.asarray(mask.flatten(), dtype=np.uint8) for mask in masks]        
        self.instances[0,-1] = 0
        self.instances[1:,-1] = np.cumsum([mask.shape[0] for mask in masks[:-1]])
        self.masks = np.concatenate(masks)

        self.shape = (self.rows.shape[0], self.cols.shape[0])
        self.dtype = np.dtype(np.int32)
        self.nbytes = self.instances.nbytes + self.rows.nbytes + self.cols.nbytes + self.masks.nbytes, gt.nbytes + inst.nbytes
        
        assert np.all(gt == self.get_semantic_image())
        assert np.all(inst == self.get_instance_image())
        
    def get_semantic_image(self):
        result = np.zeros(self.shape, dtype=self.dtype)
        for c, y0, y1, x0, x1, begin in self.instances:
            height, width = y1 - y0, x1 - x0
            end = begin + height*width
            mask = self.masks[begin:end].reshape(height, width)
            result[y0:y1,x0:x1][mask!=0] = c
        return result
    
    def get_instance_image(self):
        result = np.zeros(self.shape, dtype=self.dtype)
        for i, (c, y0, y1, x0, x1, begin) in enumerate(self.instances, 1):
            height, width = y1 - y0, x1 - x0
            end = begin + height*width
            mask = self.masks[begin:end].reshape(height, width)
            result[y0:y1,x0:x1][mask!=0] = i
        return result
