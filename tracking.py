import sys
import warnings
import torch, torchvision
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import multiprocessing

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

warnings.filterwarnings("ignore")

class Tracker:

    centers = None
    colors_ = [(int(color[2]), int(color[1]), int(color[0])) for color in [(np.array(color) * 255).astype(np.uint8) for name, color in mcolors.BASE_COLORS.items()]]
    pool_ = multiprocessing.Pool(multiprocessing.cpu_count())

    def __init__(self, threshold = 0.95, gpu = True, head_perc = 0.25, upper_body_perc = 0.15, num_clusters = 2):
        '''Initialize tracker, parameters: threshold for mask-rcnn detection (default 0.95),
        gpu True if gpu is used (default True),
        head_perc is the assumed percentage of the head's height in a human's bounding box (default 0.25),
        upper_body_perc is the assumed percentage of the upper body's height in a human's bounding box (default 0.15),
        num_clusters is the number of clusters used in kmeans clustering (default 2)'''
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        if not gpu:
            cfg.MODEL.DEVICE='cpu'
        self.predictor = DefaultPredictor(cfg)
        self.head_perc = head_perc
        self.upper_body_perc = upper_body_perc
        self.num_clusters = num_clusters

    def find_dist_(self, prev_center, new_center):
        """Find distance between two Lab color centers"""
        return 8 * np.sqrt((float(prev_center[1]) - float(new_center[1])) ** 2 + (float(prev_center[2]) - float(new_center[2])) ** 2) + 2 * abs(prev_center[0] - new_center[0])

    def find_all_dists_(self, centers):
        """Find all distances between 2 lists of color centers"""
        ret = []
        for cen_i, center in enumerate(centers):
            for c_i, c in enumerate(self.centers):
                ret.append((self.find_dist_(c, center), (cen_i, c_i)))
        return sorted(ret, key = lambda x: x[0])

    def update_all_dists_(self, all_dists, match):
        """Remove distances of centers that have already been matched"""
        return [i for i in all_dists if (i[1][0] != match[0] and i[1][1] != match[1])]

    def update_cur_center_(self, matches, center):
        """Update current center"""
        matched = []
        for match in matches:
            matched.append(match[0])
            self.centers[match[1]] = (center[match[0]] + self.centers[match[1]]) / 2
        for i in range(len(center)):
            if i not in matched:
                self.centers.append(center[i])

    def find_dom_color_(self, box, mask, image_gb):
        """Find dominant color of upper body part using kmeans clustering"""
        x1, y1, x2, y2 = box
        h = y2 - y1
        upper_body_mask = np.zeros(image_gb.shape[:2], dtype = bool)
        upper_body_mask[int(y1 + h * self.head_perc) : int(y1 + h * (self.upper_body_perc + self.head_perc)), x1 : x2] = True
        upper_body_mask = np.logical_and(mask, upper_body_mask)
        img1 = cv2.cvtColor(image_gb, cv2.COLOR_BGR2LAB)

        #kmeans to find dominant color
        data = img1[upper_body_mask]
        if len(data) >= 2:
            clt = KMeans(n_clusters = self.num_clusters, n_jobs = 2)
            clt.fit(data)
            centers_ = clt.cluster_centers_
            temp = np.unique(clt.labels_, return_counts = True)[1]
            center = list(map(lambda a: a[1], sorted(enumerate(centers_), key = lambda a: temp[a[0]], reverse = True)))[0]
            return center
        #Person mask is not eligible
        return None

    def fix_masks_(self, masks, indices):
        """Remove from every mask any common pixels with another mask"""
        new_masks = []
        for ind in indices:
            mask = masks[ind]
            for ind1 in indices:
                if ind != ind1:
                    mask = np.logical_and(mask, np.logical_and(masks[ind], np.logical_not(masks[ind1])))
            new_masks.append(mask)
        return new_masks

    def visualize_tracking_(self, img):
        """Show image with colored bounding boxes on people"""
        img1 = np.copy(img)
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box
            img1 = cv2.rectangle(img1, (x1, y1), (x2, y2), self.colors_[self.ids[i] % len(self.colors_)], 2)
        cv2.imshow("image", img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def track(self, img, visualize = False):
        outputs = self.predictor(img)
        instances = outputs["instances"].to("cpu")

        #Gaussian Blurring
        image_gb = cv2.GaussianBlur(img, (7, 7), 0)

        #Indices of detected people in outputs
        indices = np.where(instances.pred_classes.numpy() == 0)[0]
        masks = self.fix_masks_(instances.pred_masks.numpy(), indices)

        #Parallel computation of color centers
        centers = self.pool_.starmap(self.find_dom_color_, [(instances.pred_boxes.tensor[indices[ind_i]].numpy().astype(np.uint32), masks[ind_i], image_gb) for ind_i in range(len(indices))])

        new_indices = []
        new_centers = []

        for ci, center in enumerate(centers):
            if not center is None:
                new_indices.append(indices[ci])
                new_centers.append(center)

        indices = new_indices
        centers = new_centers

        self.boxes = [instances.pred_boxes.tensor[ind].numpy().astype(np.uint32) for ind in indices]
        self.masks = [instances.pred_masks.numpy()[ind] for ind in indices]

        if not self.centers:
            self.centers = centers
            self.ids = [i for i in range(len(self.centers))]
        else:
            #Calculate distances between previous and current centers
            all_dists = self.find_all_dists_(centers)
            matches = []

            #Pick minimum distance as match till no more possible matches are available
            while all_dists:
                matches.append(all_dists[0][1])
                all_dists = self.update_all_dists_(all_dists, all_dists[0][1])

            #Update center
            self.centers = self.update_cur_center_(matches, centers)
            self.ids = [-1 for i in range(len(self.boxes))]

            for match in matches:
                self.ids[match[0]] = match[1]

            marker = len(matches)
            for i, id in enumerate(self.ids):
                if id == -1:
                    self.ids[i] = marker
                    marker += 1

        if visualize:
            self.visualize_tracking_(img)
