from mrcnn.utils import compute_ap


class EvalImage():
    def __init__(self, dataset, model, cfg):
        self.dataset = dataset
        self.model = model
        self.cfg = cfg

    def evaluate_model(self, len=50):
        APs = list()
        precisions_dict = {}
        recall_dict = {}
        for index, image_id in enumerate(self.dataset.image_ids):
            print(index);
            if (index > len):
                break;
                # load image, bounding boxes and masks for the image id
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(self.dataset, self.cfg, image_id,
                                                                                      use_mini_mask=False)
            # convert pixel values (e.g. center)
            # scaled_image = modellib.mold_image(image, self.cfg)
            # convert image into one sample
            sample = np.expand_dims(image, 0)
            # print(len(image))
            # make prediction
            yhat = self.model.detect(sample, verbose=1)
            # extract results for first sample
            r = yhat[0]
            # calculate statistics, including AP
            AP, precisions, recalls, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"],
                                                    r["scores"], r['masks'])
            precisions_dict[image_id] = np.mean(precisions)
            recall_dict[image_id] = np.mean(recalls)
            # store
            APs.append(AP)

        # calculate the mean AP across all images
        mAP = np.mean(APs)
        return mAP, precisions_dict, recall_dict