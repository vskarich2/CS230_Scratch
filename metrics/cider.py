from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from constants import *

def calculate_coco_scores(o):
    if o.args.local_mode:
        annotation_file = LOCAL_COCO_CAPTION
        results_file = LOCAL_COCO_RESULTS
    else:
        annotation_file = REMOTE_COCO_CAPTION
        results_file = REMOTE_COCO_RESULTS

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')


