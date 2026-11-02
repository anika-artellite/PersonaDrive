"""
Code that loads the dataset for training.
partially taken from https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py
(MIT licence)
"""

import os
import ujson
import json
import numpy as np
import random
import cv2
import re
import gzip

import torch
from simlingo_training.utils.custom_types import DatasetOutput
from simlingo_training.dataloader.dataset_base import BaseDataset


VIZ_DATA = False


def load_icl_image(rgb_path, cut_bottom_quarter=True):
    """Load a single ICL example image from disk, applying the same crop as the main pipeline.

    Paths in icl_examples.json.gz are absolute and were captured under a
    job-scoped tmp dir; _normalize_path remaps stale paths to a real path
    on the current node.
    """
    from simlingo_training.dataloader._path_resolve import _normalize_path
    rgb_path = _normalize_path(rgb_path)
    img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if cut_bottom_quarter:
        img = img[:int(img.shape[0] - (img.shape[0] * 4.8) // 16), :, :]
    img = np.transpose(img, (2, 0, 1))  # [C, H, W]
    return img[np.newaxis]  # [1, C, H, W]


def _resolve_icl_examples(current_measurement, measurement_file_current,
                          sample_start, hist_len, use_aggregated_dataset):
    """Populate current_measurement['icl_examples'] from the per-route aggregated file.

    Looks up the per-route icl_examples.json.gz once and stores the entry for
    the current frame on the measurement dict.
    """
    if 'icl_examples' in current_measurement:
        return
    import gzip as _gzip
    if use_aggregated_dataset:
        route_dir = measurement_file_current
        frame_index = sample_start + hist_len - 1
    else:
        route_dir = os.path.dirname(os.path.dirname(measurement_file_current))
        frame_index = int(os.path.splitext(
            os.path.basename(measurement_file_current.replace('.gz', '')))[0])
    icl_agg_path = os.path.join(route_dir, 'icl_examples.json.gz')
    if not os.path.exists(icl_agg_path):
        return
    try:
        with _gzip.open(icl_agg_path, 'rt', encoding='utf-8') as _f:
            icl_agg = json.load(_f)
        for frame_key in (str(frame_index), f"{frame_index:04d}"):
            if frame_key in icl_agg:
                current_measurement['icl_examples'] = icl_agg[frame_key]
                return
    except Exception:
        pass


def build_icl_prefix(icl_examples, num_icl_examples, cut_bottom_quarter):
    """Build the ICL text prefix and load the matching images.

    Returns (icl_prefix_or_None, list_of_images_or_None). When no images
    can be loaded both are None and the caller should skip ICL.
    """
    loaded_icl = []
    icl_prefix = "Driving profile examples:\n\n"
    for ex in icl_examples[:num_icl_examples]:
        img = load_icl_image(ex['rgb_path'], cut_bottom_quarter=cut_bottom_quarter)
        if img is None:
            continue
        loaded_icl.append(img)
        speeds = ex['speed_history']
        throttles = ex['throttle_history']
        wps_str = ' '.join([f"[{round(float(w[0]),2)},{round(float(w[1]),2)}]" for w in ex['waypoints']])
        ex_block = (
            f"<image>\n"
            f"Speed: t-2={speeds[0]:.1f} t-1={speeds[1]:.1f} t0={speeds[2]:.1f} m/s. "
            f"Throttle: t-2={throttles[0]:.2f} t-1={throttles[1]:.2f} t0={throttles[2]:.2f}. "
            f"Command: {ex['command']}.\n"
        )
        tp = ex.get('target_point')
        tp_next = ex.get('target_point_next')
        if tp is not None and tp_next is not None:
            ex_block += (
                f"Target waypoint: 1:[{round(float(tp[0]),2)},{round(float(tp[1]),2)}] "
                f"2:[{round(float(tp_next[0]),2)},{round(float(tp_next[1]),2)}].\n"
            )
        ex_commentary = ex.get('commentary')
        if ex_commentary:
            ex_block += f"Commentary: {ex_commentary}\n"
        ex_block += f"Waypoints: {wps_str}\n\n"
        icl_prefix += ex_block
    if not loaded_icl:
        return None, None
    return icl_prefix, loaded_icl


class Data_Driving(BaseDataset):  # pylint: disable=locally-disabled, invalid-name
    """
    Custom dataset that dynamically loads a CARLA dataset from disk.
    """

    def __init__(self,
            **cfg,
        ):
        """Initialize the driving dataset with base config settings."""
        super().__init__(dreamer=False, **cfg)
        # Debug: log the first few commentary/QA resolutions per worker so we
        # can verify the dataloader actually finds the language data.
        self._debug_lang_logs_remaining = 3

    def __getitem__(self, index):
        """Return a single training sample for driving/QA/commentary.

        Builds prompt/answer text, loads images and waypoints, and returns
        a DatasetOutput object used by the DataModule collator.
        """
        # Disable threading because the data loader will already split in threads.
        cv2.setNumThreads(0)

        data = {}
        images = self.images[index]
        measurements = self.measurements[index]
        sample_start = self.sample_start[index]
        augment_exists = self.augment_exists[index]

        ######################################################
        ######## load current and future measurements ########
        ######################################################
        loaded_measurements, current_measurement, measurement_file_current = self.load_current_and_future_measurements(
            measurements,
            sample_start
            )
        
        data['measurement_path'] = measurement_file_current

        # Determine whether the augmented camera or the normal camera is used.
        if augment_exists and random.random() <= self.img_shift_augmentation_prob and self.img_shift_augmentation:
            augment_sample = True
            aug_rotation = current_measurement['augmentation_rotation']
            aug_translation = current_measurement['augmentation_translation']
        else:
            augment_sample = False
            aug_rotation = 0.0
            aug_translation = 0.0


        ######################################################
        ################## load waypoints ####################
        ######################################################
        #speed waypointn
        data = self.load_waypoints(data, loaded_measurements, aug_translation, aug_rotation)
       
        speed_rounded = round(current_measurement['speed'], 1)
        data['speed'] = current_measurement['speed']

        data = self.load_route(data, current_measurement, aug_translation, aug_rotation)

        target_point = np.array(current_measurement['target_point'])
        target_point = self.augment_target_point(target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
        next_target_point = np.array(current_measurement['target_point_next'])
        next_target_point = self.augment_target_point(next_target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)

        ######################################################
        ################## get commentary & qa ##################
        ######################################################
        commentary_exists = False
        commentary = ''
        if self.use_commentary:
            if self.use_aggregated_dataset:
                # measurement_file_current is the route dir itself (set in
                # load_current_and_future_measurements), so do NOT take dirname.
                route_dir = measurement_file_current
                commentary_file_path = os.path.join(route_dir.replace('data/', 'commentary/'), 'commentary.json')
            else:
                commentary_file_path = measurement_file_current.replace('measurements', 'commentary').replace('data/', 'commentary/') # TODO: move to config
            # do not use evaluation routes!!!
            if 'validation_' in commentary_file_path:
                commentary_exists = False
            else:
                try:
                    if self.use_aggregated_dataset:
                        route_dir = os.path.dirname(commentary_file_path)
                        if self.cache_dataset and route_dir in self.commentary_cache:
                            commentary_data = self.commentary_cache[route_dir]
                        else:
                            with open(commentary_file_path, 'r') as f:
                                commentary_data = ujson.load(f)
                            if self.cache_dataset:
                                self.commentary_cache[route_dir] = commentary_data

                        frame_index = sample_start + self.hist_len - 1
                        commentary_file = commentary_data[frame_index]
                    else:
                        with gzip.open(commentary_file_path, 'rt') as f:
                            commentary_file = ujson.load(f)
                    commentary_exists = commentary_file is not None
                except (FileNotFoundError, ujson.JSONDecodeError, IndexError) as _e:
                    commentary_exists = False
                    commentary_file = None
                    if self._debug_lang_logs_remaining > 0:
                        print(f"[lang-debug] commentary MISSING ({type(_e).__name__}): {commentary_file_path}", flush=True)
                if self._debug_lang_logs_remaining > 0 and commentary_exists:
                    _txt = (commentary_file.get('commentary') or '')[:120]
                    print(f"[lang-debug] commentary OK: frame={frame_index} path={commentary_file_path}\n             text='{_txt}'", flush=True)

                if commentary_file is not None:

                    commentary = commentary_file['commentary']
                    # we only augment in 60% of the cases and use the default commentary in 40% of the cases
                    # augmentation is used to increase generalization to a broader set of sentences
                    # but we do not want to overfit to the augmented sentences
                    if self.commentary_augmentation and random.random() < 0.6:
                        if commentary_file['commentary_template'] in self.templates_commentary:
                            commentary = random.choice(self.templates_commentary[commentary_file['commentary_template']])
                            for key, value in commentary_file['placeholder'].items():
                                if key in commentary:
                                    commentary = commentary.replace(key, value)
                            # regex check if <OBJECT> or <LOCATION> or any other <> is still in commentary, if so use default commentary_file['commentary']
                            if re.search(r'<.*?>', commentary):
                                print(f"WARNING: {commentary} contains placeholders that are not replaced. Using default commentary.")
                                commentary = commentary_file['commentary']

                    commentary = commentary.replace('..', '.')
                    commentary = commentary.replace('in in', 'in')
        
        qa_exists = False
        if self.use_qa:
            if self.use_aggregated_dataset:
                # measurement_file_current is the route dir itself; same fix as commentary.
                route_dir = measurement_file_current
                qa_path = os.path.join(route_dir.replace('data/', 'drivelm/'), 'vqa.json')
            else:
                qa_path = measurement_file_current.replace('measurements', 'vqa').replace('data/', 'drivelm/')
            if 'validation_' in qa_path:
                qa_exists = False
            else:
                try:
                    if self.use_aggregated_dataset:
                        route_dir = os.path.dirname(qa_path)
                        if self.cache_dataset and route_dir in self.qa_cache:
                            qa_data = self.qa_cache[route_dir]
                        else:
                            with open(qa_path, 'r') as f:
                                qa_data = ujson.load(f)
                            if self.cache_dataset:
                                self.qa_cache[route_dir] = qa_data

                        frame_index = sample_start + self.hist_len - 1
                        qa = qa_data[frame_index]
                    else:
                        with gzip.open(qa_path, 'rt') as f:
                            qa = ujson.load(f)
                    qa_exists = qa is not None
                except (FileNotFoundError, ujson.JSONDecodeError, IndexError) as _e:
                    qa_exists = False
                    qa = None
                    if self._debug_lang_logs_remaining > 0:
                        print(f"[lang-debug] qa MISSING ({type(_e).__name__}): {qa_path}", flush=True)
                if self._debug_lang_logs_remaining > 0 and qa_exists:
                    _qa_keys = list(qa.get('QA', {}).keys())[:3] if isinstance(qa, dict) else []
                    print(f"[lang-debug] qa OK: frame={frame_index} path={qa_path}\n             QA categories={_qa_keys}", flush=True)
                    self._debug_lang_logs_remaining -= 1

            if qa_exists:
                qas = qa['QA']
                qas = [values for values in qas.values()] # list of lists
                qas = [item for sublist in qas for item in sublist] # flatten list
                while True:
                    qa_chosen = random.choice(qas)
                    qa_question = qa_chosen['Q']
                    qa_answer = qa_chosen['A']
                    
                    # TODO: make this nicer!!!
                    if 'There are no pedestrians.' in qa_answer or \
                            'There is no traffic light' in qa_answer or \
                            'There are no pedestrians.' in qa_answer or \
                            'No, the ego vehicle is not affected by a stop sign.' in qa_answer or \
                            'No, the ego vehicle is not affected by a junction.' in qa_answer or \
                            'There is no traffic light affecting the ego vehicle.' in qa_answer or \
                            'There is no stop sign affecting the ego vehicle.' in qa_answer or \
                            'There is no junction affecting the ego vehicle.' in qa_answer or \
                            'It is not possible to tell' in qa_answer or \
                            'There is no reason for the ego vehicle to brake.' in qa_answer:
                        # only keep in 20% of the cases
                        if random.random() < 0.2:
                            break
                    else:
                        break
                
                # we only augment in 60% of the cases and use the default QA in 40% of the cases
                # augmentation is used to increase generalization to a broader set of sentences
                # but we do not want to overfit to the augmented sentences
                if self.qa_augmentation and random.random() < 0.6:
                    qa_question_org = qa_question
                    qa_answer_org = qa_answer
                    locations = [
                        'nearby to the front of the ego vehicle',
                        'nearby to the front right of the ego vehicle',
                        'nearby to the front left of the ego vehicle',
                        'nearby on the left side of the ego vehicle',
                        'far to the front left of the ego vehicle',
                        'far to the front right of the ego vehicle',
                        'far to the front of the ego vehicle',
                        'far to the left side of the ego vehicle',
                        'far to the right side of the ego vehicle',
                        'to the front of the ego vehicle',
                        'to the front right of the ego vehicle',
                        'to the front left of the ego vehicle',
                        'on the left side of the ego vehicle',
                        'on the right side of the ego vehicle',
                    ]
                    objects = [value['Visual_description'] for key, value in qa['key_object_infos'].items()]
                    q_objects = []
                    a_objects = []
                    for object_type in objects:
                        if object_type in qa_question:
                            qa_question = qa_question.replace(object_type, '<OBJECT>')
                            q_objects.append(object_type)
                        if object_type in qa_answer:
                            qa_answer = qa_answer.replace(object_type, '<OBJECT>')
                            a_objects.append(object_type)
                    
                    q_location = ''
                    a_location = ''
                    for location in locations:
                        if location in qa_question:
                            qa_question = qa_question.replace(location, '<LOCATION>')
                            q_location = location
                        if location in qa_answer:
                            qa_answer = qa_answer.replace(location, '<LOCATION>')
                            a_location = location
                        
                    q_distance = re.search(r'in (\d+) m', qa_question)
                    qa_question = re.sub(r'in \d+ m', 'in <DISTANCE>', qa_question)
                    a_distance = re.search(r'in (\d+) m', qa_answer)
                    qa_answer = re.sub(r'in \d+ m', 'in <DISTANCE>', qa_answer)
                    if len(q_objects)==0:
                        q_objects = ['']
                    if len(a_objects)==0:
                        a_objects = ['']
                    
                    # in 40% of the cases we do not augment the question
                    if len(q_objects) > 1 or len(a_objects) > 1 or random.random() < 0.4: 
                        qa_question = qa_question_org
                        qa_answer = qa_answer_org
                    else:
                        if qa_question in self.q_augment:
                            qa_question = random.choice(self.q_augment[qa_question]).replace('<OBJECT>', q_objects[0]).replace('<LOCATION>', q_location)
                            if q_distance:
                                qa_question = qa_question.replace('<DISTANCE>', q_distance.group(1))
                        else:
                            print(f"WARNING: {qa_question} not in q_augment. Using default question.")
                            qa_question = qa_question_org
                        if qa_answer in self.a_augment:
                            qa_answer = random.choice(self.a_augment[qa_answer]).replace('<OBJECT>', a_objects[0]).replace('<LOCATION>', a_location)
                            if a_distance:
                                qa_answer = qa_answer.replace('<DISTANCE>', a_distance.group(1))
                        else:
                            print(f"WARNING: {qa_answer} not in a_augment. Using default answer.")
                            qa_answer = qa_answer_org

        ######################################################
        ######## load navigational_conditioning ########
        ######################################################
        target_options, placeholder_values = self.get_navigational_conditioning( data, current_measurement, target_point, next_target_point)
            
        answer = ''

        prompt_random = random.random()
        
        if self.use_commentary and commentary_exists and prompt_random < self.prompt_probabilities['commentary']:
            if random.random() < 0.2: # 20% of the time we give commentary as prompt
                if random.random() < 0.5:
                    prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} {commentary} Predict the waypoints."
                else:
                    prompt = f"Current speed: {speed_rounded} m/s. Command: {commentary} Predict the waypoints."
                answer = f"Waypoints:"
            else:
                # 80% of the time we want to predict commentary
                prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} What should the ego do next?"
                answer = f"{commentary} Waypoints:"
            self.num_sampled_per_type['commentary'] += 1
            
        elif self.use_qa and qa_exists and prompt_random < (self.prompt_probabilities['qa'] + self.prompt_probabilities['commentary']):
            prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} Q: {qa_question}"
            answer = f"A: {qa_answer}"
            self.num_sampled_per_type['qa'] += 1
            
        else:
            prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} Predict the waypoints."
            answer = f"Waypoints:"
            self.num_sampled_per_type['driving'] += 1

        # recalculate the probabilties after warmup (when more than 1000 samples have been sampled)
        # we do this in case we dont have qa or commentary for every sample otherwise it would lead to undersampling one of those
        if sum(self.num_sampled_per_type.values()) > 10000 and sum(self.num_sampled_per_type.values()) % 10000 == 0:
            self.prompt_probabilities = {key: (1/value if value > 0 else 0.0) for key, value in self.num_sampled_per_type.items()}
            total = sum(self.prompt_probabilities.values())
            if total > 0:
                self.prompt_probabilities = {key: value/total for key, value in self.prompt_probabilities.items()}
            print(f"Prompt probabilities: {self.prompt_probabilities}")
            print(f"Number of samples per type: {self.num_sampled_per_type}")
            
        answer = answer.replace('..', '.')
        prompt = prompt.replace('..', '.')

        ######################################################
        ######## load current and past images ########
        ######################################################
        data = self.load_images(data, images, augment_sample=augment_sample)

        ######################################################
        ######## build ICL prefix (driving profile examples) ########
        ######################################################
        icl_images = None
        use_icl = getattr(self, 'use_icl', False)
        num_icl_examples = getattr(self, 'num_icl_examples', 2)

        if use_icl:
            _resolve_icl_examples(
                current_measurement, measurement_file_current,
                sample_start, self.hist_len, self.use_aggregated_dataset,
            )
        if use_icl and 'icl_examples' in current_measurement:
            # Match load_images crop predicate: shift-augment also requires
            # bonnet removal, so the query is cropped whenever either flag is
            # set. ICL frames must follow the same rule or np.concatenate in
            # the collate fails on mismatched H.
            icl_prefix, loaded_icl = build_icl_prefix(
                current_measurement['icl_examples'],
                num_icl_examples,
                self.cut_bottom_quarter or self.img_shift_augmentation,
            )
            if loaded_icl:
                icl_images = loaded_icl
                prompt = icl_prefix + "Current scene:\n<image>\n" + prompt

        conversation_answer = [
            {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"{answer}"},
                ],
            },
        ]
        conversation_all = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}"},
                {"type": "image"},
                ],
            },
            {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"{answer}"},
                ],
            }
        ]

        data_new = DatasetOutput(
            conversation = conversation_all,
            answer = conversation_answer,
            image_ff = data['rgb'],
            image_ff_org_size=data['rgb_org_size'],
            waypoints = data["waypoints"],
            waypoints_1d = data["waypoints_1d"],
            path = data['route_adjusted'],
            target_points = data['target_points'],
            speed = data['speed'],
            placeholder_values = placeholder_values,
            measurement_path = data['measurement_path'],
            dataset = 'driving',
            icl_images = icl_images,
        )
        
        if VIZ_DATA:
            # front image with path and waypoints and commentary
            self.visualise_cameras(data_new, commentary, data['route_adjusted'], data['waypoints'], options=None, prompt=prompt, answer=answer, name="img")
        return data_new


if __name__ == "__main__":
    from hydra import compose, initialize
    from simlingo_training.config import TrainConfig
    
    # seed all
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    

    initialize(config_path="../config")
    cfg = compose(config_name="config")
    
    cfg.data_module.base_dataset.use_commentary = True
    cfg.data_module.base_dataset.use_qa = True
    cfg.data_module.base_dataset.img_shift_augmentation = False

    print('Test Dataset')
    dataset = Data_Driving(                        
                        split="train",
                        bucket_name='all',
                        **cfg.data_module,
                        **cfg.data_module.base_dataset,
    )

    for i in range(len(dataset)):
        # shuffle
        # i = np.random.randint(0, len(dataset))
        data = dataset[i]
        # print(data)
        # if i == 100:
        #     break