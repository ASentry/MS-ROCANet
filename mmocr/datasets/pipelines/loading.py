import mmcv
import numpy as np
from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.loading import LoadAnnotations, LoadImageFromFile


@PIPELINES.register_module()
class LoadTextAnnotations(LoadAnnotations):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True):
        super().__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=poly2mask)

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.

        Args:
            polygons (list[list]): Polygons of one instance.

        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p).astype(np.float32) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        ann_info = results['ann_info']
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = ann_info['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        gt_masks_ignore = ann_info.get('masks_ignore', None)
        if gt_masks_ignore is not None:
            if self.poly2mask:
                gt_masks_ignore = BitmapMasks(
                    [self._poly2mask(mask, h, w) for mask in gt_masks_ignore],
                    h, w)
            else:
                gt_masks_ignore = PolygonMasks([
                    self.process_polygons(polygons)
                    for polygons in gt_masks_ignore
                ], h, w)
            results['gt_masks_ignore'] = gt_masks_ignore
            results['mask_fields'].append('gt_masks_ignore')

        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results


@PIPELINES.register_module()
class LoadImageFromNdarray(LoadImageFromFile):
    """Load an image from np.ndarray.

    Similar with :obj:`LoadImageFromFile`, but the image read from
    ``results['img']``, which is np.ndarray.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        assert results['img'].dtype == 'uint8'

        img = results['img']
        if self.color_type == 'grayscale' and img.shape[2] == 3:
            img = mmcv.bgr2gray(img, keepdim=True)
        if self.color_type == 'color' and img.shape[2] == 1:
            img = mmcv.gray2bgr(img)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results
