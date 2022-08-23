from mmcv import Config
from mmdet.apis import async_inference_detector, inference_detector

from .ssod.apis.inference import init_detector
from .ssod.utils import patch_config


async def async_inference(config, checkpoint, device, imgs):
    cfg = Config.fromfile(config)
    # Not affect anything, just avoid index error
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, checkpoint, device=device)
    # imgs = glob.glob(imgs)
    results = []
    for img in imgs:
        # test a single image
        result = await async_inference_detector(model, img)
        results.append(result)
    return results


def inference(config, checkpoint, device, imgs):
    cfg = Config.fromfile(config)
    # Not affect anything, just avoid index error
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, checkpoint, device=device)
    # imgs = glob.glob(imgs)
    results = []
    for img in imgs:
        # test a single image
        result = inference_detector(model, img)
        results.append(result)
    return results
