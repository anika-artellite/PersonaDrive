"""
Code that loads the dataset for training.
partially taken from https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py
(MIT licence)
"""
import json
import os
import ujson
import numpy as np
import random
import cv2
import re
import gzip

from simlingo_training.utils.custom_types import DatasetOutput
from simlingo_training.dataloader.dataset_base import BaseDataset
from simlingo_training.dataloader.dataset_driving import (
    build_icl_prefix,
    load_icl_image,
    _resolve_icl_examples,
)

class Data_Eval(BaseDataset):  # pylint: disable=locally-disabled, invalid-name
    """
    Custom dataset that dynamically loads a CARLA dataset from disk.
    """

    def __init__(self,
            **cfg,
        ):
        """Initialize the QA/commentary evaluation dataset."""
        super().__init__(dreamer=False, evaluation=True, **cfg)

    def __getitem__(self, index):
        """Return a single QA/commentary evaluation sample.

        Uses preselected evaluation questions or commentary and returns
        DatasetOutput with the chosen QA template if available.
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
        # In aggregated mode, measurement_file_current is the route dir;
        # the per-route commentary.json/vqa.json is a list indexed by frame.
        eval_key = (
            (measurement_file_current, sample_start + self.hist_len - 1)
            if self.use_aggregated_dataset
            else measurement_file_current
        )
        frame_index = sample_start + self.hist_len - 1

        commentary_exists = False
        commentary = ''
        if self.use_commentary:
            if self.use_aggregated_dataset:
                commentary_file_path = os.path.join(
                    measurement_file_current.replace('data/', 'commentary/'),
                    'commentary.json',
                )
            else:
                commentary_file_path = measurement_file_current.replace('measurements', 'commentary').replace('data/', 'commentary/') # TODO: move to config
            try:
                if self.use_aggregated_dataset:
                    with open(commentary_file_path, 'r') as f:
                        commentary_data = ujson.load(f)
                    commentary_file = commentary_data[frame_index]
                else:
                    with gzip.open(commentary_file_path, 'rt') as f:
                        commentary_file = ujson.load(f)
                commentary_exists = commentary_file is not None
            except (FileNotFoundError, ujson.JSONDecodeError, IndexError):
                commentary_exists = False
                commentary_file = None

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
        qa_chosen = None
        if self.use_qa:
            if self.use_aggregated_dataset:
                qa_path = os.path.join(
                    measurement_file_current.replace('data/', 'drivelm/'),
                    'vqa.json',
                )
            else:
                qa_path = measurement_file_current.replace('measurements', 'vqa').replace('data/', 'drivelm/')
            try:
                if self.use_aggregated_dataset:
                    with open(qa_path, 'r') as f:
                        qa_data = ujson.load(f)
                    qa = qa_data[frame_index]
                else:
                    with gzip.open(qa_path, 'rt') as f:
                        qa = ujson.load(f)
                qa_exists = qa is not None
            except (FileNotFoundError, ujson.JSONDecodeError, IndexError):
                qa_exists = False
                qa = None

            if qa_exists:
                qas = qa['QA']
                qas = [values for values in qas.values()] # list of lists
                qas = [item for sublist in qas for item in sublist] # flatten list
                questions = [qa['Q'] for qa in qas]
                answers = [qa['A'] for qa in qas]

                qa_chosen = self.all_eval_samples_dict[eval_key]
                qa_question = qa_chosen[0][0]
                qa_answer = qa_chosen[0][1]

                if qa_question in questions and qa_answer in answers:
                    pass
                elif '<OBJECT>' in qa_question or '<LOCATION>' in qa_question or '<DISTANCE>' in qa_question or '<OBJECT>' in qa_answer or '<LOCATION>' in qa_answer or '<DISTANCE>' in qa_answer:
                    for question, answer in zip(questions, answers):
                        question_org = question
                        answer_org = answer
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
                            'nearby',
                            'far',
                        ]
                        objects = [value['Visual_description'] for key, value in qa['key_object_infos'].items()]
                        q_objects = []
                        a_objects = []
                        for object_type in objects:
                            if object_type in question:
                                question = question.replace(object_type, '<OBJECT>')
                                q_objects.append(object_type)
                            if object_type in answer:
                                answer = answer.replace(object_type, '<OBJECT>')
                                a_objects.append(object_type)
                        
                        q_location = ''
                        a_location = ''
                        for location in locations:
                            if location in question:
                                question = question.replace(location, '<LOCATION>')
                                q_location = location
                            if location in answer:
                                answer = answer.replace(location, '<LOCATION>')
                                a_location = location
                            
                        q_distance = re.search(r'in (\d+) m', question)
                        question = re.sub(r'in \d+ m', 'in <DISTANCE>', question)
                        a_distance = re.search(r'in (\d+) m', answer)
                        answer = re.sub(r'in \d+ m', 'in <DISTANCE>', answer)
                        
                        if qa_question == question and qa_answer == answer:
                            qa_question = question_org
                            qa_answer = answer_org
                            break

        ######################################################
        ######## load navigational_conditioning ########
        ######################################################
        target_options, placeholder_values = self.get_navigational_conditioning( data, current_measurement, target_point, next_target_point)
            
        answer = ''

        if self.use_commentary:
            prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} What should the ego do next?"
            answer = f"{commentary} Waypoints:"
            
        elif self.use_qa and qa_exists:
            prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} Q: {qa_question}"
            answer = f"A: {qa_answer}"
        else:
            raise ValueError(f"Neither commentary nor qa exists. {commentary_exists}, {qa_exists}")
            
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
            # Match load_images crop predicate so ICL frames have the same H
            # as the (possibly shift-aug-cropped) query frame.
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
        
        images = [data['rgb']]

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
            qa_templates = qa_chosen,
            icl_images = icl_images,
        )
        
        return data_new