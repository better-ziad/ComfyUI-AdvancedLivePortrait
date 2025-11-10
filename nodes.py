import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
import folder_paths
import comfy.utils
import time
import copy
import dill
import yaml
from ultralytics import YOLO

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

from .LivePortrait.live_portrait_wrapper import LivePortraitWrapper
from .LivePortrait.utils.camera import get_rotation_matrix
from .LivePortrait.config.inference_config import InferenceConfig

from .LivePortrait.modules.spade_generator import SPADEDecoder
from .LivePortrait.modules.warping_network import WarpingNetwork
from .LivePortrait.modules.motion_extractor import MotionExtractor
from .LivePortrait.modules.appearance_feature_extractor import AppearanceFeatureExtractor
from .LivePortrait.modules.stitching_retargeting_network import StitchingRetargetingNetwork
from collections import OrderedDict

cur_device = None
def get_device():
    global cur_device
    if cur_device == None:
        if torch.cuda.is_available():
            cur_device = torch.device('cuda')
            print("Uses CUDA device.")
        elif torch.backends.mps.is_available():
            cur_device = torch.device('mps')
            print("Uses MPS device.")
        else:
            cur_device = torch.device('cpu')
            print("Uses CPU device.")
    return cur_device

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
def rgb_crop(rgb, region):
    return rgb[region[1]:region[3], region[0]:region[2]]

def rgb_crop_batch(rgbs, region):
    return rgbs[:, region[1]:region[3], region[0]:region[2]]
def get_rgb_size(rgb):
    return rgb.shape[1], rgb.shape[0]
def create_transform_matrix(x, y, s_x, s_y):
    return np.float32([[s_x, 0, x], [0, s_y, y]])

def get_model_dir(m):
    try:
        return folder_paths.get_folder_paths(m)[0]
    except:
        return os.path.join(folder_paths.models_dir, m)

def calc_crop_limit(center, img_size, crop_size):
    pos = center - crop_size / 2
    if pos < 0:
        crop_size += pos * 2
        pos = 0

    pos2 = pos + crop_size

    if img_size < pos2:
        crop_size -= (pos2 - img_size) * 2
        pos2 = img_size
        pos = pos2 - crop_size

    return pos, pos2, crop_size

def retargeting(delta_out, driving_exp, factor, idxes):
    for idx in idxes:
        #delta_out[0, idx] -= src_exp[0, idx] * factor
        delta_out[0, idx] += driving_exp[0, idx] * factor

class PreparedSrcImg:
    def __init__(self, src_rgb, crop_trans_m, x_s_info, f_s_user, x_s_user, mask_ori):
        self.src_rgb = src_rgb
        self.crop_trans_m = crop_trans_m
        self.x_s_info = x_s_info
        self.f_s_user = f_s_user
        self.x_s_user = x_s_user
        self.mask_ori = mask_ori

import requests
from tqdm import tqdm

class LP_Engine:
    pipeline = None
    detect_model = None
    mask_img = None
    temp_img_idx = 0

    def get_temp_img_name(self):
        self.temp_img_idx += 1
        return "expression_edit_preview" + str(self.temp_img_idx) + ".png"

    def download_model(_, file_path, model_url):
        print('AdvancedLivePortrait: Downloading model...')
        response = requests.get(model_url, stream=True)
        try:
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte

                # tqdm will display a progress bar
                with open(file_path, 'wb') as file, tqdm(
                        desc='Downloading',
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(block_size):
                        bar.update(len(data))
                        file.write(data)

        except requests.exceptions.RequestException as err:
            print('AdvancedLivePortrait: Model download failed: {err}')
            print(f'AdvancedLivePortrait: Download it manually from: {model_url}')
            print(f'AdvancedLivePortrait: And put it in {file_path}')
        except Exception as e:
            print(f'AdvancedLivePortrait: An unexpected error occurred: {e}')

    def remove_ddp_dumplicate_key(_, state_dict):
        state_dict_new = OrderedDict()
        for key in state_dict.keys():
            state_dict_new[key.replace('module.', '')] = state_dict[key]
        return state_dict_new

    def filter_for_model(_, checkpoint, prefix):
        filtered_checkpoint = {key.replace(prefix + "_module.", ""): value for key, value in checkpoint.items() if
                               key.startswith(prefix)}
        return filtered_checkpoint

    def load_model(self, model_config, model_type):

        device = get_device()

        if model_type == 'stitching_retargeting_module':
            ckpt_path = os.path.join(get_model_dir("liveportrait"), "retargeting_models", model_type + ".pth")
        else:
            ckpt_path = os.path.join(get_model_dir("liveportrait"), "base_models", model_type + ".pth")

        is_safetensors = None
        if os.path.isfile(ckpt_path) == False:
            is_safetensors = True
            ckpt_path = os.path.join(get_model_dir("liveportrait"), model_type + ".safetensors")
            if os.path.isfile(ckpt_path) == False:
                self.download_model(ckpt_path,
                "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/" + model_type + ".safetensors")
        model_params = model_config['model_params'][f'{model_type}_params']
        if model_type == 'appearance_feature_extractor':
            model = AppearanceFeatureExtractor(**model_params).to(device)
        elif model_type == 'motion_extractor':
            model = MotionExtractor(**model_params).to(device)
        elif model_type == 'warping_module':
            model = WarpingNetwork(**model_params).to(device)
        elif model_type == 'spade_generator':
            model = SPADEDecoder(**model_params).to(device)
        elif model_type == 'stitching_retargeting_module':
            # Special handling for stitching and retargeting module
            config = model_config['model_params']['stitching_retargeting_module_params']
            checkpoint = comfy.utils.load_torch_file(ckpt_path)

            stitcher = StitchingRetargetingNetwork(**config.get('stitching'))
            if is_safetensors:
                stitcher.load_state_dict(self.filter_for_model(checkpoint, 'retarget_shoulder'))
            else:
                stitcher.load_state_dict(self.remove_ddp_dumplicate_key(checkpoint['retarget_shoulder']))
            stitcher = stitcher.to(device)
            stitcher.eval()

            return {
                'stitching': stitcher,
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")


        model.load_state_dict(comfy.utils.load_torch_file(ckpt_path))
        model.eval()
        return model

    def load_models(self):
        model_path = get_model_dir("liveportrait")
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        model_config_path = os.path.join(current_directory, 'LivePortrait', 'config', 'models.yaml')
        model_config = yaml.safe_load(open(model_config_path, 'r'))

        appearance_feature_extractor = self.load_model(model_config, 'appearance_feature_extractor')
        motion_extractor = self.load_model(model_config, 'motion_extractor')
        warping_module = self.load_model(model_config, 'warping_module')
        spade_generator = self.load_model(model_config, 'spade_generator')
        stitching_retargeting_module = self.load_model(model_config, 'stitching_retargeting_module')

        self.pipeline = LivePortraitWrapper(InferenceConfig(), appearance_feature_extractor, motion_extractor, warping_module, spade_generator, stitching_retargeting_module)

    def get_detect_model(self):
        if self.detect_model == None:
            model_dir = get_model_dir("ultralytics")
            if not os.path.exists(model_dir): os.mkdir(model_dir)
            model_path = os.path.join(model_dir, "face_yolov8n.pt")
            if not os.path.exists(model_path):
                self.download_model(model_path, "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt")
            self.detect_model = YOLO(model_path)

        return self.detect_model

    def get_face_bboxes(self, image_rgb):
        detect_model = self.get_detect_model()
        pred = detect_model(image_rgb, conf=0.7, device="")
        return pred[0].boxes.xyxy.cpu().numpy()

    def detect_face(self, image_rgb, crop_factor, sort = True):
        bboxes = self.get_face_bboxes(image_rgb)
        w, h = get_rgb_size(image_rgb)

        print(f"w, h:{w, h}")

        cx = w / 2
        min_diff = w
        best_box = None
        for x1, y1, x2, y2 in bboxes:
            bbox_w = x2 - x1
            if bbox_w < 30: continue
            diff = abs(cx - (x1 + bbox_w / 2))
            if diff < min_diff:
                best_box = [x1, y1, x2, y2]
                print(f"diff, min_diff, best_box:{diff, min_diff, best_box}")
                min_diff = diff

        if best_box == None:
            print("Failed to detect face!!")
            return [0, 0, w, h]

        x1, y1, x2, y2 = best_box

        #for x1, y1, x2, y2 in bboxes:
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        crop_w = bbox_w * crop_factor
        crop_h = bbox_h * crop_factor

        crop_w = max(crop_h, crop_w)
        crop_h = crop_w

        kernel_x = int(x1 + bbox_w / 2)
        kernel_y = int(y1 + bbox_h / 2)

        new_x1 = int(kernel_x - crop_w / 2)
        new_x2 = int(kernel_x + crop_w / 2)
        new_y1 = int(kernel_y - crop_h / 2)
        new_y2 = int(kernel_y + crop_h / 2)

        if not sort:
            return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

        if new_x1 < 0:
            new_x2 -= new_x1
            new_x1 = 0
        elif w < new_x2:
            new_x1 -= (new_x2 - w)
            new_x2 = w
            if new_x1 < 0:
                new_x2 -= new_x1
                new_x1 = 0

        if new_y1 < 0:
            new_y2 -= new_y1
            new_y1 = 0
        elif h < new_y2:
            new_y1 -= (new_y2 - h)
            new_y2 = h
            if new_y1 < 0:
                new_y2 -= new_y1
                new_y1 = 0

        if w < new_x2 and h < new_y2:
            over_x = new_x2 - w
            over_y = new_y2 - h
            over_min = min(over_x, over_y)
            new_x2 -= over_min
            new_y2 -= over_min

        return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]


    def calc_face_region(self, square, dsize):
        region = copy.deepcopy(square)
        is_changed = False
        if dsize[0] < region[2]:
            region[2] = dsize[0]
            is_changed = True
        if dsize[1] < region[3]:
            region[3] = dsize[1]
            is_changed = True

        return region, is_changed

    def expand_img(self, rgb_img, square):
        #new_img = rgb_crop(rgb_img, face_region)
        crop_trans_m = create_transform_matrix(max(-square[0], 0), max(-square[1], 0), 1, 1)
        new_img = cv2.warpAffine(rgb_img, crop_trans_m, (square[2] - square[0], square[3] - square[1]),
                                        cv2.INTER_LINEAR)
        return new_img

    def get_pipeline(self):
        if self.pipeline == None:
            print("Load pipeline...")
            self.load_models()

        return self.pipeline

    def prepare_src_image(self, img):
        h, w = img.shape[:2]
        input_shape = [256,256]
        if h != input_shape[0] or w != input_shape[1]:
            if 256 < h: interpolation = cv2.INTER_AREA
            else: interpolation = cv2.INTER_LINEAR
            x = cv2.resize(img, (input_shape[0], input_shape[1]), interpolation = interpolation)
        else:
            x = img.copy()

        if x.ndim == 3:
            x = x[np.newaxis].astype(np.float32) / 255.  # HxWx3 -> 1xHxWx3, normalized to 0~1
        elif x.ndim == 4:
            x = x.astype(np.float32) / 255.  # BxHxWx3, normalized to 0~1
        else:
            raise ValueError(f'img ndim should be 3 or 4: {x.ndim}')
        x = np.clip(x, 0, 1)  # clip to 0~1
        x = torch.from_numpy(x).permute(0, 3, 1, 2)  # 1xHxWx3 -> 1x3xHxW
        x = x.to(get_device())
        return x

    def GetMaskImg(self):
        if self.mask_img is None:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./LivePortrait/utils/resources/mask_template.png")
            self.mask_img = cv2.imread(path, cv2.IMREAD_COLOR)
        return self.mask_img

    def crop_face(self, img_rgb, crop_factor):
        crop_region = self.detect_face(img_rgb, crop_factor)
        face_region, is_changed = self.calc_face_region(crop_region, get_rgb_size(img_rgb))
        face_img = rgb_crop(img_rgb, face_region)
        if is_changed: face_img = self.expand_img(face_img, crop_region)
        return face_img

    def prepare_source(self, source_image, crop_factor, is_video = False, tracking = False):
        print("Prepare source...")
        engine = self.get_pipeline()
        source_image_np = (source_image * 255).byte().numpy()
        img_rgb = source_image_np[0]

        psi_list = []
        for img_rgb in source_image_np:
            if tracking or len(psi_list) == 0:
                crop_region = self.detect_face(img_rgb, crop_factor)
                face_region, is_changed = self.calc_face_region(crop_region, get_rgb_size(img_rgb))

                s_x = (face_region[2] - face_region[0]) / 512.
                s_y = (face_region[3] - face_region[1]) / 512.
                crop_trans_m = create_transform_matrix(crop_region[0], crop_region[1], s_x, s_y)
                mask_ori = cv2.warpAffine(self.GetMaskImg(), crop_trans_m, get_rgb_size(img_rgb), cv2.INTER_LINEAR)
                mask_ori = mask_ori.astype(np.float32) / 255.

                if is_changed:
                    s = (crop_region[2] - crop_region[0]) / 512.
                    crop_trans_m = create_transform_matrix(crop_region[0], crop_region[1], s, s)

            face_img = rgb_crop(img_rgb, face_region)
            if is_changed: face_img = self.expand_img(face_img, crop_region)
            i_s = self.prepare_src_image(face_img)
            x_s_info = engine.get_kp_info(i_s)
            f_s_user = engine.extract_feature_3d(i_s)
            x_s_user = engine.transform_keypoint(x_s_info)
            psi = PreparedSrcImg(img_rgb, crop_trans_m, x_s_info, f_s_user, x_s_user, mask_ori)
            if is_video == False:
                return psi
            psi_list.append(psi)

        return psi_list

    def prepare_driving_video(self, face_images):
        print("Prepare driving video...")
        pipeline = self.get_pipeline()
        f_img_np = (face_images * 255).byte().numpy()

        out_list = []
        for f_img in f_img_np:
            i_d = self.prepare_src_image(f_img)
            d_info = pipeline.get_kp_info(i_d)
            out_list.append(d_info)

        return out_list

    def calc_fe(_, x_d_new, eyes, eyebrow, wink, pupil_x, pupil_y, mouth, eee, woo, smile,
                rotate_pitch, rotate_yaw, rotate_roll):

        x_d_new[0, 20, 1] += smile * -0.01
        x_d_new[0, 14, 1] += smile * -0.02
        x_d_new[0, 17, 1] += smile * 0.0065
        x_d_new[0, 17, 2] += smile * 0.003
        x_d_new[0, 13, 1] += smile * -0.00275
        x_d_new[0, 16, 1] += smile * -0.00275
        x_d_new[0, 3, 1] += smile * -0.0035
        x_d_new[0, 7, 1] += smile * -0.0035

        x_d_new[0, 19, 1] += mouth * 0.001
        x_d_new[0, 19, 2] += mouth * 0.0001
        x_d_new[0, 17, 1] += mouth * -0.0001
        rotate_pitch -= mouth * 0.05

        x_d_new[0, 20, 2] += eee * -0.001
        x_d_new[0, 20, 1] += eee * -0.001
        #x_d_new[0, 19, 1] += eee * 0.0006
        x_d_new[0, 14, 1] += eee * -0.001

        x_d_new[0, 14, 1] += woo * 0.001
        x_d_new[0, 3, 1] += woo * -0.0005
        x_d_new[0, 7, 1] += woo * -0.0005
        x_d_new[0, 17, 2] += woo * -0.0005

        x_d_new[0, 11, 1] += wink * 0.001
        x_d_new[0, 13, 1] += wink * -0.0003
        x_d_new[0, 17, 0] += wink * 0.0003
        x_d_new[0, 17, 1] += wink * 0.0003
        x_d_new[0, 3, 1] += wink * -0.0003
        rotate_roll -= wink * 0.1
        rotate_yaw -= wink * 0.1

        if 0 < pupil_x:
            x_d_new[0, 11, 0] += pupil_x * 0.0007
            x_d_new[0, 15, 0] += pupil_x * 0.001
        else:
            x_d_new[0, 11, 0] += pupil_x * 0.001
            x_d_new[0, 15, 0] += pupil_x * 0.0007

        x_d_new[0, 11, 1] += pupil_y * -0.001
        x_d_new[0, 15, 1] += pupil_y * -0.001
        eyes -= pupil_y / 2.

        x_d_new[0, 11, 1] += eyes * -0.001
        x_d_new[0, 13, 1] += eyes * 0.0003
        x_d_new[0, 15, 1] += eyes * -0.001
        x_d_new[0, 16, 1] += eyes * 0.0003
        x_d_new[0, 1, 1] += eyes * -0.00025
        x_d_new[0, 2, 1] += eyes * 0.00025


        if 0 < eyebrow:
            x_d_new[0, 1, 1] += eyebrow * 0.001
            x_d_new[0, 2, 1] += eyebrow * -0.001
        else:
            x_d_new[0, 1, 0] += eyebrow * -0.001
            x_d_new[0, 2, 0] += eyebrow * 0.001
            x_d_new[0, 1, 1] += eyebrow * 0.0003
            x_d_new[0, 2, 1] += eyebrow * -0.0003


        return torch.Tensor([rotate_pitch, rotate_yaw, rotate_roll])
g_engine = LP_Engine()

class ExpressionSet:
    def __init__(self, erst = None, es = None):
        if es != None:
            self.e = copy.deepcopy(es.e)  # [:, :, :]
            self.r = copy.deepcopy(es.r)  # [:]
            self.s = copy.deepcopy(es.s)
            self.t = copy.deepcopy(es.t)
        elif erst != None:
            self.e = erst[0]
            self.r = erst[1]
            self.s = erst[2]
            self.t = erst[3]
        else:
            self.e = torch.from_numpy(np.zeros((1, 21, 3))).float().to(get_device())
            self.r = torch.Tensor([0, 0, 0])
            self.s = 0
            self.t = 0
    def div(self, value):
        self.e /= value
        self.r /= value
        self.s /= value
        self.t /= value
    def add(self, other):
        self.e += other.e
        self.r += other.r
        self.s += other.s
        self.t += other.t
    def sub(self, other):
        self.e -= other.e
        self.r -= other.r
        self.s -= other.s
        self.t -= other.t
    def mul(self, value):
        self.e *= value
        self.r *= value
        self.s *= value
        self.t *= value

    #def apply_ratio(self, ratio):        self.exp *= ratio

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time - start_time))
        return result

    return wrapper_fn


#exp_data_dir = os.path.join(current_directory, "exp_data")
exp_data_dir = os.path.join(folder_paths.output_directory, "exp_data")
if os.path.isdir(exp_data_dir) == False:
    os.mkdir(exp_data_dir)
class SaveExpData:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "file_name": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {"save_exp": ("EXP_DATA",), }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_name",)
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"
    OUTPUT_NODE = True

    def run(self, file_name, save_exp:ExpressionSet=None):
        if save_exp == None or file_name == "":
            return file_name

        with open(os.path.join(exp_data_dir, file_name + ".exp"), "wb") as f:
            dill.dump(save_exp, f)

        return file_name

class LoadExpData:
    @classmethod
    def INPUT_TYPES(s):
        file_list = [os.path.splitext(file)[0] for file in os.listdir(exp_data_dir) if file.endswith('.exp')]
        return {"required": {
            "file_name": (sorted(file_list, key=str.lower),),
            "ratio": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
        },
        }

    RETURN_TYPES = ("EXP_DATA",)
    RETURN_NAMES = ("exp",)
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"

    def run(self, file_name, ratio):
        # es = ExpressionSet()
        with open(os.path.join(exp_data_dir, file_name + ".exp"), 'rb') as f:
            es = dill.load(f)
        es.mul(ratio)
        return (es,)

class ExpData:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
                #"code": ("STRING", {"multiline": False, "default": ""}),
                "code1": ("INT", {"default": 0}),
                "value1": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.1}),
                "code2": ("INT", {"default": 0}),
                "value2": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.1}),
                "code3": ("INT", {"default": 0}),
                "value3": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.1}),
                "code4": ("INT", {"default": 0}),
                "value4": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.1}),
                "code5": ("INT", {"default": 0}),
                "value5": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.1}),
            },
            "optional":{"add_exp": ("EXP_DATA",),}
        }

    RETURN_TYPES = ("EXP_DATA",)
    RETURN_NAMES = ("exp",)
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"

    def run(self, code1, value1, code2, value2, code3, value3, code4, value4, code5, value5, add_exp=None):
        if add_exp == None:
            es = ExpressionSet()
        else:
            es = ExpressionSet(es = add_exp)

        codes = [code1, code2, code3, code4, code5]
        values = [value1, value2, value3, value4, value5]
        for i in range(5):
            idx = int(codes[i] / 10)
            r = codes[i] % 10
            es.e[0, idx, r] += values[i] * 0.001

        return (es,)

class PrintExpData:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "cut_noise": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 0.1}),
        },
            "optional": {"exp": ("EXP_DATA",), }
        }

    RETURN_TYPES = ("EXP_DATA",)
    RETURN_NAMES = ("exp",)
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"
    OUTPUT_NODE = True

    def run(self, cut_noise, exp = None):
        if exp == None: return (exp,)

        cuted_list = []
        e = exp.exp * 1000
        for idx in range(21):
            for r in range(3):
                a = abs(e[0, idx, r])
                if(cut_noise < a): cuted_list.append((a, e[0, idx, r], idx*10+r))

        sorted_list = sorted(cuted_list, reverse=True, key=lambda item: item[0])
        print(f"sorted_list: {[[item[2], round(float(item[1]),1)] for item in sorted_list]}")
        return (exp,)

class HeadSizeControl:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "head_size": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.01}),
            "neck_scale": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 1.5, "step": 0.01}),
        },
            "optional": {"add_exp": ("EXP_DATA",), }
        }

    RETURN_TYPES = ("EXP_DATA",)
    RETURN_NAMES = ("exp",)
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"

    def run(self, head_size, neck_scale, add_exp=None):
        if add_exp == None:
            es = ExpressionSet()
        else:
            es = ExpressionSet(es=add_exp)

        es.s = head_size
        
        # Apply neck scaling if different from 1.0  
        if neck_scale != 1.0:
            # Use similar magnitude to facial expressions (0.001 to 0.01 range)
            neck_adjustment = (neck_scale - 1.0) * 0.02
            
            # Target unused keypoints that might control jaw/neck area
            # Avoid: 1,2(eyebrows), 3,7(cheeks), 11,13,15,16(eyes), 14,17,19,20(mouth)
            # Try: 0, 4, 5, 6, 8, 9, 10, 12, 18 (potential jaw/neck keypoints)
            
            # Primary neck width control - X axis adjustments
            jaw_keypoints = [4, 5, 6, 8, 9, 10, 12, 18]
            
            for idx in jaw_keypoints:
                # Adjust width (X coordinate) 
                es.e[0, idx, 0] += neck_adjustment
                
            # If available, try keypoint 0 for central control
            if neck_adjustment != 0:
                es.e[0, 0, 0] += neck_adjustment * 0.5
                es.e[0, 0, 1] += neck_adjustment * 0.3

        return (es,)

class Command:
    def __init__(self, es, change, keep):
        self.es:ExpressionSet = es
        self.change = change
        self.keep = keep

crop_factor_default = 1.7
crop_factor_min = 1.5
crop_factor_max = 2.5

class AdvancedLivePortrait:
    def __init__(self):
        self.src_images = None
        self.driving_images = None
        self.pbar = comfy.utils.ProgressBar(1)
        self.crop_factor = None

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "retargeting_eyes": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
                "retargeting_mouth": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
                "crop_factor": ("FLOAT", {"default": crop_factor_default,
                                          "min": crop_factor_min, "max": crop_factor_max, "step": 0.1}),
                "turn_on": ("BOOLEAN", {"default": True}),
                "tracking_src_vid": ("BOOLEAN", {"default": False}),
                "animate_without_vid": ("BOOLEAN", {"default": False}),
                "command": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "src_images": ("IMAGE",),
                "motion_link": ("EDITOR_LINK",),
                "driving_images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "AdvancedLivePortrait"

    # INPUT_IS_LIST = False
    # OUTPUT_IS_LIST = (False,)

    def parsing_command(self, command, motoin_link):
        command.replace(' ', '')
        # if command == '': return
        lines = command.split('\n')

        cmd_list = []

        total_length = 0

        i = 0
        #old_es = None
        for line in lines:
            i += 1
            if line == '': continue
            try:
                cmds = line.split('=')
                idx = int(cmds[0])
                if idx == 0: es = ExpressionSet()
                else: es = ExpressionSet(es = motoin_link[idx])
                cmds = cmds[1].split(':')
                change = int(cmds[0])
                keep = int(cmds[1])
            except:
                assert False, f"(AdvancedLivePortrait) Command Err Line {i}: {line}"


                return None, None

            total_length += change + keep
            es.div(change)
            cmd_list.append(Command(es, change, keep))

        return cmd_list, total_length


    def run(self, retargeting_eyes, retargeting_mouth, turn_on, tracking_src_vid, animate_without_vid, command, crop_factor,
            src_images=None, driving_images=None, motion_link=None):
        if turn_on == False: return (None,None)
        src_length = 1

        if src_images == None:
            if motion_link != None:
                self.psi_list = [motion_link[0]]
            else: return (None,None)

        if src_images != None:
            src_length = len(src_images)
            if id(src_images) != id(self.src_images) or self.crop_factor != crop_factor:
                self.crop_factor = crop_factor
                self.src_images = src_images
                if 1 < src_length:
                    self.psi_list = g_engine.prepare_source(src_images, crop_factor, True, tracking_src_vid)
                else:
                    self.psi_list = [g_engine.prepare_source(src_images, crop_factor)]


        cmd_list, cmd_length = self.parsing_command(command, motion_link)
        if cmd_list == None: return (None,None)
        cmd_idx = 0

        driving_length = 0
        if driving_images is not None:
            if id(driving_images) != id(self.driving_images):
                self.driving_images = driving_images
                self.driving_values = g_engine.prepare_driving_video(driving_images)
            driving_length = len(self.driving_values)

        total_length = max(driving_length, src_length)

        if animate_without_vid:
            total_length = max(total_length, cmd_length)

        c_i_es = ExpressionSet()
        c_o_es = ExpressionSet()
        d_0_es = None
        out_list = []

        psi = None
        pipeline = g_engine.get_pipeline()
        for i in range(total_length):

            if i < src_length:
                psi = self.psi_list[i]
                s_info = psi.x_s_info
                s_es = ExpressionSet(erst=(s_info['kp'] + s_info['exp'], torch.Tensor([0, 0, 0]), s_info['scale'], s_info['t']))

            new_es = ExpressionSet(es = s_es)

            if i < cmd_length:
                cmd = cmd_list[cmd_idx]
                if 0 < cmd.change:
                    cmd.change -= 1
                    c_i_es.add(cmd.es)
                    c_i_es.sub(c_o_es)
                elif 0 < cmd.keep:
                    cmd.keep -= 1

                new_es.add(c_i_es)

                if cmd.change == 0 and cmd.keep == 0:
                    cmd_idx += 1
                    if cmd_idx < len(cmd_list):
                        c_o_es = ExpressionSet(es = c_i_es)
                        cmd = cmd_list[cmd_idx]
                        c_o_es.div(cmd.change)
            elif 0 < cmd_length:
                new_es.add(c_i_es)

            if i < driving_length:
                d_i_info = self.driving_values[i]
                d_i_r = torch.Tensor([d_i_info['pitch'], d_i_info['yaw'], d_i_info['roll']])#.float().to(device="cuda:0")

                if d_0_es is None:
                    d_0_es = ExpressionSet(erst = (d_i_info['exp'], d_i_r, d_i_info['scale'], d_i_info['t']))

                    retargeting(s_es.e, d_0_es.e, retargeting_eyes, (11, 13, 15, 16))
                    retargeting(s_es.e, d_0_es.e, retargeting_mouth, (14, 17, 19, 20))

                new_es.e += d_i_info['exp'] - d_0_es.e
                new_es.r += d_i_r - d_0_es.r
                new_es.t += d_i_info['t'] - d_0_es.t

            r_new = get_rotation_matrix(
                s_info['pitch'] + new_es.r[0], s_info['yaw'] + new_es.r[1], s_info['roll'] + new_es.r[2])
            d_new = new_es.s * (new_es.e @ r_new) + new_es.t
            d_new = pipeline.stitching(psi.x_s_user, d_new)
            crop_out = pipeline.warp_decode(psi.f_s_user, psi.x_s_user, d_new)
            crop_out = pipeline.parse_output(crop_out['out'])[0]

            crop_with_fullsize = cv2.warpAffine(crop_out, psi.crop_trans_m, get_rgb_size(psi.src_rgb),
                                                cv2.INTER_LINEAR)
            out = np.clip(psi.mask_ori * crop_with_fullsize + (1 - psi.mask_ori) * psi.src_rgb, 0, 255).astype(
                np.uint8)
            out_list.append(out)

            self.pbar.update_absolute(i+1, total_length, ("PNG", Image.fromarray(crop_out), None))

        if len(out_list) == 0: return (None,)

        out_imgs = torch.cat([pil2tensor(img_rgb) for img_rgb in out_list])
        return (out_imgs,)

class ExpressionEditor:
    def __init__(self):
        self.sample_image = None
        self.src_image = None
        self.crop_factor = None

    @classmethod
    def INPUT_TYPES(s):
        display = "number"
        #display = "slider"
        return {
            "required": {

                "rotate_pitch": ("FLOAT", {"default": 0, "min": -20, "max": 20, "step": 0.5, "display": display}),
                "rotate_yaw": ("FLOAT", {"default": 0, "min": -20, "max": 20, "step": 0.5, "display": display}),
                "rotate_roll": ("FLOAT", {"default": 0, "min": -20, "max": 20, "step": 0.5, "display": display}),

                "blink": ("FLOAT", {"default": 0, "min": -20, "max": 5, "step": 0.5, "display": display}),
                "eyebrow": ("FLOAT", {"default": 0, "min": -10, "max": 15, "step": 0.5, "display": display}),
                "wink": ("FLOAT", {"default": 0, "min": 0, "max": 25, "step": 0.5, "display": display}),
                "pupil_x": ("FLOAT", {"default": 0, "min": -15, "max": 15, "step": 0.5, "display": display}),
                "pupil_y": ("FLOAT", {"default": 0, "min": -15, "max": 15, "step": 0.5, "display": display}),
                "aaa": ("FLOAT", {"default": 0, "min": -30, "max": 120, "step": 1, "display": display}),
                "eee": ("FLOAT", {"default": 0, "min": -20, "max": 15, "step": 0.2, "display": display}),
                "woo": ("FLOAT", {"default": 0, "min": -20, "max": 15, "step": 0.2, "display": display}),
                "smile": ("FLOAT", {"default": 0, "min": -0.3, "max": 1.3, "step": 0.01, "display": display}),

                "src_ratio": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01, "display": display}),
                "sample_ratio": ("FLOAT", {"default": 1, "min": -0.2, "max": 1.2, "step": 0.01, "display": display}),
                "sample_parts": (["OnlyExpression", "OnlyRotation", "OnlyMouth", "OnlyEyes", "All"],),
                "crop_factor": ("FLOAT", {"default": crop_factor_default,
                                          "min": crop_factor_min, "max": crop_factor_max, "step": 0.1}),
            },

            "optional": {"src_image": ("IMAGE",), "motion_link": ("EDITOR_LINK",),
                         "sample_image": ("IMAGE",), "add_exp": ("EXP_DATA",),
            },
        }

    RETURN_TYPES = ("IMAGE", "EDITOR_LINK", "EXP_DATA")
    RETURN_NAMES = ("image", "motion_link", "save_exp")

    FUNCTION = "run"

    OUTPUT_NODE = True

    CATEGORY = "AdvancedLivePortrait"

    # INPUT_IS_LIST = False
    # OUTPUT_IS_LIST = (False,)

    def run(self, rotate_pitch, rotate_yaw, rotate_roll, blink, eyebrow, wink, pupil_x, pupil_y, aaa, eee, woo, smile,
            src_ratio, sample_ratio, sample_parts, crop_factor, src_image=None, sample_image=None, motion_link=None, add_exp=None):
        rotate_yaw = -rotate_yaw

        new_editor_link = None
        if motion_link != None:
            self.psi = motion_link[0]
            new_editor_link = motion_link.copy()
        elif src_image != None:
            if id(src_image) != id(self.src_image) or self.crop_factor != crop_factor:
                self.crop_factor = crop_factor
                self.psi = g_engine.prepare_source(src_image, crop_factor)
                self.src_image = src_image
            new_editor_link = []
            new_editor_link.append(self.psi)
        else:
            return (None,None)

        pipeline = g_engine.get_pipeline()

        psi = self.psi
        s_info = psi.x_s_info
        #delta_new = copy.deepcopy()
        s_exp = s_info['exp'] * src_ratio
        s_exp[0, 5] = s_info['exp'][0, 5]
        s_exp += s_info['kp']

        es = ExpressionSet()

        if sample_image != None:
            if id(self.sample_image) != id(sample_image):
                self.sample_image = sample_image
                d_image_np = (sample_image * 255).byte().numpy()
                d_face = g_engine.crop_face(d_image_np[0], 1.7)
                i_d = g_engine.prepare_src_image(d_face)
                self.d_info = pipeline.get_kp_info(i_d)
                self.d_info['exp'][0, 5, 0] = 0
                self.d_info['exp'][0, 5, 1] = 0

            # "OnlyExpression", "OnlyRotation", "OnlyMouth", "OnlyEyes", "All"
            if sample_parts == "OnlyExpression" or sample_parts == "All":
                es.e += self.d_info['exp'] * sample_ratio
            if sample_parts == "OnlyRotation" or sample_parts == "All":
                rotate_pitch += self.d_info['pitch'] * sample_ratio
                rotate_yaw += self.d_info['yaw'] * sample_ratio
                rotate_roll += self.d_info['roll'] * sample_ratio
            elif sample_parts == "OnlyMouth":
                retargeting(es.e, self.d_info['exp'], sample_ratio, (14, 17, 19, 20))
            elif sample_parts == "OnlyEyes":
                retargeting(es.e, self.d_info['exp'], sample_ratio, (1, 2, 11, 13, 15, 16))

        es.r = g_engine.calc_fe(es.e, blink, eyebrow, wink, pupil_x, pupil_y, aaa, eee, woo, smile,
                                  rotate_pitch, rotate_yaw, rotate_roll)

        if add_exp != None:
            es.add(add_exp)

        new_rotate = get_rotation_matrix(s_info['pitch'] + es.r[0], s_info['yaw'] + es.r[1],
                                         s_info['roll'] + es.r[2])
        x_d_new = (s_info['scale'] * (1 + es.s)) * ((s_exp + es.e) @ new_rotate) + s_info['t']

        x_d_new = pipeline.stitching(psi.x_s_user, x_d_new)

        crop_out = pipeline.warp_decode(psi.f_s_user, psi.x_s_user, x_d_new)
        crop_out = pipeline.parse_output(crop_out['out'])[0]

        crop_with_fullsize = cv2.warpAffine(crop_out, psi.crop_trans_m, get_rgb_size(psi.src_rgb), cv2.INTER_LINEAR)
        out = np.clip(psi.mask_ori * crop_with_fullsize + (1 - psi.mask_ori) * psi.src_rgb, 0, 255).astype(np.uint8)

        out_img = pil2tensor(out)

        filename = g_engine.get_temp_img_name() #"fe_edit_preview.png"
        folder_paths.get_save_image_path(filename, folder_paths.get_temp_directory())
        img = Image.fromarray(crop_out)
        img.save(os.path.join(folder_paths.get_temp_directory(), filename), compress_level=1)
        results = list()
        results.append({"filename": filename, "type": "temp"})

        new_editor_link.append(es)

        return {"ui": {"images": results}, "result": (out_img, new_editor_link, es)}

class NeckSlimming:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_image": ("IMAGE",),
                "face_mask": ("IMAGE",),
                "neck_mask": ("IMAGE",),
                "thin_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.5, "step": 0.1}),
                "feather_px": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 50.0, "step": 1.0}),
                "dilate_px": ("INT", {"default": 6, "min": 0, "max": 20, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("slimmed_image",)
    FUNCTION = "slim_neck"
    CATEGORY = "AdvancedLivePortrait"

    def slim_neck(self, face_image, face_mask, neck_mask, thin_factor, feather_px, dilate_px):
        # Convert from ComfyUI tensors to numpy arrays
        img_np = (face_image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        face_mask_np = (face_mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        neck_mask_np = (neck_mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Handle mask dimensions - take the first channel if RGB
        if len(face_mask_np.shape) == 3:
            face_mask_gray = face_mask_np[:, :, 0]
        else:
            face_mask_gray = face_mask_np
            
        if len(neck_mask_np.shape) == 3:
            neck_mask_gray = neck_mask_np[:, :, 0]
        else:
            neck_mask_gray = neck_mask_np

        h, w = neck_mask_gray.shape[:2]
        M = (neck_mask_gray > 127).astype(np.uint8)
        F = (face_mask_gray > 127).astype(np.uint8)

        # Dilate face mask to ensure no warping near jawline
        if dilate_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
            F_dil = cv2.dilate(F, kernel)
        else:
            F_dil = F

        # Distance transforms
        inside_dt = cv2.distanceTransform(M, cv2.DIST_L2, 3)
        outside_dt = cv2.distanceTransform(1 - M, cv2.DIST_L2, 3)

        inside_power = 1.0
        outside_scale = 0.25

        inside_w = (inside_dt / (inside_dt.max() + 1e-6)) ** inside_power if inside_dt.max() > 0 else inside_dt
        outside_w = np.exp(-outside_dt / max(feather_px, 1e-6))

        # Zero out weights in/near the face region to prevent any influence there
        inside_w[F_dil == 1] = 0.0
        outside_w[F_dil == 1] = 0.0

        # Build row-wise centerline over the neck (and extend to all rows)
        centers = np.full(h, np.nan, dtype=np.float32)
        ys = np.where(M.any(axis=1))[0]
        if ys.size > 0:
            for y in ys:
                xs = np.where(M[y] > 0)[0]
                if xs.size > 0:
                    centers[y] = (xs.min() + xs.max()) / 2.0
            first, last = ys.min(), ys.max()
            for y in range(first + 1, last + 1):
                if np.isnan(centers[y]): 
                    centers[y] = centers[y - 1]
            for y in range(last - 1, first - 1, -1):
                if np.isnan(centers[y]): 
                    centers[y] = centers[y + 1]
            centers[:first] = centers[first]
            centers[last+1:] = centers[last]
        else:
            centers[:] = w / 2.0

        X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        C = centers[:, None].astype(np.float32)

        dx_in = -thin_factor * (X - C) * inside_w
        dx_out = outside_scale * thin_factor * (X - C) * outside_w
        dx = dx_in + dx_out

        # Explicitly zero displacement in the (dilated) face area for extra safety
        dx[F_dil == 1] = 0.0

        map_x = np.clip(X + dx, 0, w - 1).astype(np.float32)
        map_y = Y.astype(np.float32)

        warped = cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Paste back original face pixels (non-interpolated) for exact preservation
        face_region = (F_dil == 1)
        warped[face_region] = img_bgr[face_region]

        # Convert back to RGB and ComfyUI tensor format
        result_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        result_tensor = torch.from_numpy(result_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (result_tensor,)

class HeadNeckResize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_image": ("IMAGE",),
                "head_mask": ("IMAGE",),
                "neck_mask": ("IMAGE",),
                "neck_scale_horizontal": ("FLOAT", {"default": 0.8, "min": 0.5, "max": 1.2, "step": 0.01}),
                "neck_scale_vertical": ("FLOAT", {"default": 0.8, "min": 0.5, "max": 1.2, "step": 0.01}),
                "blur_radius": ("INT", {"default": 15, "min": 5, "max": 50, "step": 1}),
                "feather_edge": ("INT", {"default": 10, "min": 2, "max": 30, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resized_image",)
    FUNCTION = "resize_head_neck"
    CATEGORY = "AdvancedLivePortrait"

    def resize_head_neck(self, face_image, head_mask, neck_mask, neck_scale_horizontal, neck_scale_vertical, blur_radius, feather_edge):
        # Convert from ComfyUI tensors to numpy arrays
        img_np = (face_image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        head_mask_np = (head_mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        neck_mask_np = (neck_mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        
        # Handle mask dimensions - take the first channel if RGB
        if len(head_mask_np.shape) == 3:
            head_mask_gray = head_mask_np[:, :, 0]
        else:
            head_mask_gray = head_mask_np
            
        if len(neck_mask_np.shape) == 3:
            neck_mask_gray = neck_mask_np[:, :, 0]
        else:
            neck_mask_gray = neck_mask_np

        h, w, c = img_np.shape
        H = (head_mask_gray > 127).astype(np.uint8)
        N = (neck_mask_gray > 127).astype(np.uint8)
        
        # Start with original image
        result = img_np.copy()
        
        # Step 1: Process neck resize if needed
        neck_position_changed = False
        original_neck_top = None
        new_neck_top = None
        
        if np.any(N > 0):
            neck_coords = np.where(N > 0)
            original_neck_top = neck_coords[0].min()
            original_neck_bottom = neck_coords[0].max()
            print(f"Original neck: top={original_neck_top}, bottom={original_neck_bottom}")
            
            # Only resize neck if scaling is different from 1.0
            if neck_scale_horizontal != 1.0 or neck_scale_vertical != 1.0:
                print(f"Resizing neck: h_scale={neck_scale_horizontal}, v_scale={neck_scale_vertical}")
                
                # Get neck bounding box
                y_min, y_max = original_neck_top, original_neck_bottom
                x_min, x_max = neck_coords[1].min(), neck_coords[1].max()
                
                # Calculate new neck dimensions
                original_neck_height = y_max - y_min + 1
                original_neck_width = x_max - x_min + 1
                new_neck_height = max(1, int(original_neck_height * neck_scale_vertical))
                new_neck_width = max(1, int(original_neck_width * neck_scale_horizontal))
                
                # New neck position (keep bottom, adjust top)
                new_neck_top = y_max - new_neck_height + 1
                neck_position_changed = True
                
                print(f"Neck resize: {original_neck_width}x{original_neck_height} -> {new_neck_width}x{new_neck_height}")
                print(f"Neck position: top {original_neck_top} -> {new_neck_top}")
                
                # Apply neck resize
                result = self._resize_region_simple(result, N, neck_scale_horizontal, neck_scale_vertical, blur_radius, feather_edge, anchor_bottom=True)
            else:
                new_neck_top = original_neck_top
        
        # # Step 2: Move head if neck position changed
        # if np.any(H > 0) and neck_position_changed and new_neck_top is not None:
        #     head_coords = np.where(H > 0)
        #     original_head_bottom = head_coords[0].max()
        #     head_height = original_head_bottom - head_coords[0].min() + 1
            
        #     # Calculate how much to move head down
        #     head_move_distance = new_neck_top - original_head_bottom
        #     print(f"Moving head down by {head_move_distance} pixels to connect with neck")
            
        #     if head_move_distance != 0:
        #         result = self._move_region(result, H, 0, head_move_distance, blur_radius, feather_edge)
        
        # Convert back to ComfyUI tensor format
        result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (result_tensor,)

    def _process_region(self, img, mask, scale, blur_radius, feather_edge, anchor_bottom=False, return_position=False, target_bottom_y=None):
        """
        Process a region (head or neck) with scaling and blending
        
        Args:
            img: Input image (H, W, C)
            mask: Binary mask for the region
            scale: Scaling factor (can be tuple for (horizontal, vertical) or single value)
            blur_radius: Blur radius for seamless blending
            feather_edge: Edge feathering for smooth transition
            anchor_bottom: If True, anchor the bottom of the region (for neck)
            return_position: If True, return the new top position
            target_bottom_y: If provided, position the region so its bottom aligns with this Y coordinate
        """
        h, w = mask.shape
        
        # Find bounding box of the region
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            if return_position:
                return img, None
            return img
            
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Extract the region
        region_h = y_max - y_min + 1
        region_w = x_max - x_min + 1
        
        # Validate region dimensions
        if region_h <= 0 or region_w <= 0:
            print(f"Warning: Invalid region dimensions: {region_w}x{region_h}")
            if return_position:
                return img, None
            return img
        
        # Handle scaling (can be tuple or single value)
        if isinstance(scale, tuple):
            scale_w, scale_h = scale
        else:
            scale_w = scale_h = scale
        
        # Calculate new dimensions
        new_h = max(1, int(region_h * scale_h))  # Ensure minimum size of 1
        new_w = max(1, int(region_w * scale_w))  # Ensure minimum size of 1
        
        # Extract and resize the image region
        img_region = img[y_min:y_max+1, x_min:x_max+1]
        mask_region = mask[y_min:y_max+1, x_min:x_max+1]
        
        # Validate extracted regions
        if img_region.size == 0 or mask_region.size == 0:
            print(f"Warning: Empty region extracted: img_region.shape={img_region.shape}, mask_region.shape={mask_region.shape}")
            if return_position:
                return img, None
            return img
        
        # Resize both image and mask
        resized_img = cv2.resize(img_region, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        resized_mask = cv2.resize(mask_region.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized_mask = (resized_mask > 0.5).astype(np.uint8)
        
        # Calculate placement position
        if target_bottom_y is not None:
            # Position head so its bottom edge aligns with target_bottom_y (top of neck)
            new_y = target_bottom_y - new_h
            new_x = x_min + (region_w - new_w) // 2
            print(f"Positioning head: target_bottom_y={target_bottom_y}, new_h={new_h}, new_y={new_y}")
        elif anchor_bottom:
            # For neck: keep bottom edge at same position, but update top
            new_y = y_max - new_h + 1
            new_x = x_min + (region_w - new_w) // 2
            print(f"Positioning neck: y_max={y_max}, new_h={new_h}, new_y={new_y}")
        else:
            # For head: center the region (fallback)
            new_y = y_min + (region_h - new_h) // 2
            new_x = x_min + (region_w - new_w) // 2
        
        # Ensure bounds
        new_y = max(0, min(h - new_h, new_y))
        new_x = max(0, min(w - new_w, new_x))
        
        # Create result image
        result = img.copy()
        
        # First, fill the original area with background estimation
        original_mask_3d = np.stack([mask] * 3, axis=-1)
        result = self._fill_background(result, original_mask_3d, blur_radius)
        
        # Create feathered mask for smooth blending
        feathered_mask = self._create_feathered_mask(resized_mask, feather_edge)
        feathered_mask_3d = np.stack([feathered_mask] * 3, axis=-1)
        
        # Place the resized region
        y_end = min(h, new_y + new_h)
        x_end = min(w, new_x + new_w)
        
        # Handle cropping if the resized region exceeds image bounds
        crop_h = y_end - new_y
        crop_w = x_end - new_x
        
        if crop_h > 0 and crop_w > 0:
            resized_img_crop = resized_img[:crop_h, :crop_w]
            feathered_mask_crop = feathered_mask_3d[:crop_h, :crop_w]
            
            # Blend the resized region
            result[new_y:y_end, new_x:x_end] = (
                resized_img_crop * feathered_mask_crop + 
                result[new_y:y_end, new_x:x_end] * (1 - feathered_mask_crop)
            ).astype(np.uint8)
        
        if return_position:
            # For neck, return the top Y position of the resized region
            return result, new_y
        return result

    def _fill_background(self, img, mask_3d, blur_radius):
        """Fill masked areas with blurred surrounding content"""
        result = img.copy()
        
        # Create a version with the masked area filled by inpainting
        mask_single = mask_3d[:, :, 0]
        
        if np.any(mask_single > 0):
            # Use OpenCV inpainting to fill the hole
            inpainted = cv2.inpaint(img, mask_single, blur_radius, cv2.INPAINT_TELEA)
            
            # Blend with blurred version for smoother result
            blurred = cv2.GaussianBlur(img, (blur_radius*2+1, blur_radius*2+1), blur_radius/3)
            
            # Use inpainted result where mask is, original where not
            result = np.where(mask_3d > 0, inpainted, img)
            
        return result

    def _create_feathered_mask(self, mask, feather_edge):
        """Create a feathered mask for smooth blending"""
        if feather_edge <= 0:
            return mask.astype(np.float32)
        
        # Distance transform for smooth falloff
        mask_float = mask.astype(np.float32)
        
        # Erode mask to create inner area
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather_edge*2+1, feather_edge*2+1))
        inner_mask = cv2.erode(mask, kernel)
        
        # Distance transform from edge
        edge_mask = mask - inner_mask
        if np.any(edge_mask > 0):
            # Create smooth gradient
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            edge_dist = cv2.distanceTransform(1 - edge_mask, cv2.DIST_L2, 3)
            
            # Normalize and create smooth falloff
            edge_dist = np.minimum(edge_dist, feather_edge)
            smooth_mask = edge_dist / feather_edge
            
            # Combine inner (full) and edge (gradient) areas
            result = np.where(inner_mask > 0, 1.0, smooth_mask)
            result = np.where(mask == 0, 0.0, result)
        else:
            result = mask_float
        
        return np.clip(result, 0.0, 1.0)

    def _resize_region_simple(self, img, mask, scale_x, scale_y, blur_radius, feather_edge, anchor_bottom=False):
        """Simple region resize with clear logic"""
        h, w = mask.shape
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return img
            
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Extract region
        region_h = y_max - y_min + 1
        region_w = x_max - x_min + 1
        new_h = max(1, int(region_h * scale_y))
        new_w = max(1, int(region_w * scale_x))
        
        img_region = img[y_min:y_max+1, x_min:x_max+1]
        mask_region = mask[y_min:y_max+1, x_min:x_max+1]
        
        # Resize
        resized_img = cv2.resize(img_region, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        resized_mask = cv2.resize(mask_region.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized_mask = (resized_mask > 0.5).astype(np.uint8)
        
        # Position
        if anchor_bottom:
            new_y = y_max - new_h + 1
        else:
            new_y = y_min + (region_h - new_h) // 2
        new_x = x_min + (region_w - new_w) // 2
        
        # Bounds check
        new_y = max(0, min(h - new_h, new_y))
        new_x = max(0, min(w - new_w, new_x))
        
        # Apply to image
        result = img.copy()
        
        # Fill original area
        original_mask_3d = np.stack([mask] * 3, axis=-1)
        result = self._fill_background(result, original_mask_3d, blur_radius)
        
        # Place resized region
        feathered_mask = self._create_feathered_mask(resized_mask, feather_edge)
        feathered_mask_3d = np.stack([feathered_mask] * 3, axis=-1)
        
        y_end = min(h, new_y + new_h)
        x_end = min(w, new_x + new_w)
        crop_h = y_end - new_y
        crop_w = x_end - new_x
        
        if crop_h > 0 and crop_w > 0:
            resized_img_crop = resized_img[:crop_h, :crop_w]
            feathered_mask_crop = feathered_mask_3d[:crop_h, :crop_w]
            result[new_y:y_end, new_x:x_end] = (
                resized_img_crop * feathered_mask_crop + 
                result[new_y:y_end, new_x:x_end] * (1 - feathered_mask_crop)
            ).astype(np.uint8)
        
        return result

    def _move_region(self, img, mask, move_x, move_y, blur_radius, feather_edge):
        """Move a region without resizing"""
        h, w = mask.shape
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return img
            
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Extract region
        img_region = img[y_min:y_max+1, x_min:x_max+1]
        mask_region = mask[y_min:y_max+1, x_min:x_max+1]
        
        # New position
        new_y = y_min + move_y
        new_x = x_min + move_x
        region_h = y_max - y_min + 1
        region_w = x_max - x_min + 1
        
        # Bounds check
        new_y = max(0, min(h - region_h, new_y))
        new_x = max(0, min(w - region_w, new_x))
        
        # Apply to image
        result = img.copy()
        
        # Fill original area
        original_mask_3d = np.stack([mask] * 3, axis=-1)
        result = self._fill_background(result, original_mask_3d, blur_radius)
        
        # Place moved region
        feathered_mask = self._create_feathered_mask(mask_region, feather_edge)
        feathered_mask_3d = np.stack([feathered_mask] * 3, axis=-1)
        
        y_end = min(h, new_y + region_h)
        x_end = min(w, new_x + region_w)
        crop_h = y_end - new_y
        crop_w = x_end - new_x
        
        if crop_h > 0 and crop_w > 0:
            img_crop = img_region[:crop_h, :crop_w]
            mask_crop = feathered_mask_3d[:crop_h, :crop_w]
            result[new_y:y_end, new_x:x_end] = (
                img_crop * mask_crop + 
                result[new_y:y_end, new_x:x_end] * (1 - mask_crop)
            ).astype(np.uint8)
        
        return result

NODE_CLASS_MAPPINGS = {
    "AdvancedLivePortrait": AdvancedLivePortrait,
    "ExpressionEditor": ExpressionEditor,
    "LoadExpData": LoadExpData,
    "SaveExpData": SaveExpData,
    "ExpData": ExpData,
    "PrintExpData:": PrintExpData,
    "HeadSizeControl": HeadSizeControl,
    "NeckSlimming": NeckSlimming,
    "HeadNeckResize": HeadNeckResize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedLivePortrait": "Advanced Live Portrait (PHM)",
    "ExpressionEditor": "Expression Editor (PHM)",
    "LoadExpData": "Load Exp Data (PHM)",
    "SaveExpData": "Save Exp Data (PHM)",
    "HeadSizeControl": "Head Size Control (PHM)",
    "NeckSlimming": "Neck Slimming (PHM)",
    "HeadNeckResize": "Head & Neck Resize (PHM)"
}