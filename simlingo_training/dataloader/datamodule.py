"""Lightning DataModule for SimLingo datasets and collation logic."""

# Standard library imports
import itertools
from typing import List

# Third-party imports
import hydra
import line_profiler
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoProcessor

# Local/project specific imports
# from simlingo_training.dataloader.dataset_driving import Data_Driving # is called directly by hydra.utils.instantiate, keeping here to make it easier to find
# from simlingo_training.dataloader.dataset_dreamer import Data_Dreamer # is called directly by hydra.utils.instantiate, keeping here to make it easier to find
from simlingo_training.utils.custom_types import DrivingExample, DrivingInput, DrivingLabel, LanguageLabel
from simlingo_training.utils.hf_revision import hf_revision
from simlingo_training.utils.internvl2_utils import preprocess_image_batch, get_custom_chat_template, get_num_image_tokens_per_patch
from simlingo_training.utils.projection import get_camera_intrinsics, get_camera_extrinsics

def encode_uint8(strings: List[str], common_length: int) -> torch.Tensor:
    """Encode a list of strings as a fixed-length uint8 tensor.

    Args:
        strings: List of strings to encode.
        common_length: Fixed byte length for each string (pads with nulls).

    Returns:
        Tensor of shape [B, common_length] with UTF-8 bytes.
    """
    max_len = max(len(s) for s in strings)
    assert max_len <= common_length, f"String is too long: {max_len} > {common_length}"
    padded_strings = [s.ljust(common_length, '\0') for s in strings]
    return torch.tensor([bytearray(s, 'utf-8') for s in padded_strings], dtype=torch.uint8)


class DataModule(LightningDataModule):
    """Builds datasets, samplers, and dataloaders for training and eval."""
    def __init__(
        self,
        base_dataset,
        processor,
        predict=False,
        **cfg,
    ):
        """Initialize the data module with Hydra-configured datasets.

        Args:
            base_dataset: Base dataset config dict from Hydra.
            processor: Vision/LLM processor used for tokenization.
            predict: If True, only build predict dataset.
            **cfg: Remaining Hydra config fields for the data module.
        """
        super().__init__()
        for key, value in cfg.items():
            setattr(self, key, value)
            
        for key, value in base_dataset.items():
            setattr(self, key, value)
            
        self.cfg = cfg
        self.base_dataset = base_dataset
        self.processor = processor
        self.predict = predict
        
        self.printed = False

        # Each camera image is split into a 1x2 grid -> 2 patches of 448x448 pixels each.
        # The ViT inside InternVL2 uses a 14x14 patch size with a downsample ratio of 0.5,
        # which produces 256 feature vectors (tokens) per 448x448 patch.
        # So one camera image -> 2 patches x 256 tokens = 512 <IMG_CONTEXT> tokens in the text.
        # With ICL enabled (e.g. 2 examples + 1 query) -> 3 images x 512 = 1536 <IMG_CONTEXT> tokens.
        self.NUM_IMAGE_PATCHES = 2                          # number of patches per image (1x2 grid)
        self.IMAGES_TO_CONSIDER = ['image_ff']              # only front-forward camera is used

        # Special tokens that wrap image features in the text sequence:
        #   <img>...<IMG_CONTEXT>xN...</img>
        # The vision encoder fills each <IMG_CONTEXT> slot with one patch feature vector.
        # Reference: https://github.com/OpenGVLab/InternVL (internvl_chat constants + modeling)
        self.IMG_START_TOKEN='<img>'                        # marks start of image region in text
        self.IMG_END_TOKEN='</img>'                         # marks end of image region in text
        self.IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'              # one slot per patch feature vector

        self.num_image_tokens_per_patch = get_num_image_tokens_per_patch(self.encoder_variant)  # 256 for InternVL2
        self.num_image_tokens_total = self.num_image_tokens_per_patch * self.NUM_IMAGE_PATCHES  # 512 per image
            
        # add <WAYPOINT> token
        if 'tokenizer' in self.processor.__dict__:
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor
        # TODO: not needed anymore?
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<WAYPOINTS>','<WAYPOINTS_DIFF>', '<ORG_WAYPOINTS_DIFF>', '<ORG_WAYPOINTS>', '<WAYPOINT_LAST>', '<ROUTE>', '<ROUTE_DIFF>', '<TARGET_POINT>']})
        self.tokenizer.padding_side = "left"

    def setup(self, stage=None):
        """Instantiate datasets for train/val or predict mode.

        Args:
            stage: Optional Lightning stage hint (unused here).

        Behavior:
            - Train/val mode: builds bucketed datasets, sampling weights,
              a weighted sampler, and concatenated validation datasets.
            - Predict mode: builds a single evaluation dataset.

        Side effects:
            Sets `train_dataset`, `val_dataset`, `predict_dataset`, and
            (for training) `sampler_train`.
        """
        if not self.predict:
            self.val_datasets = []  # collect validation datasets to concat
            sum_sample_weights = 1.0  # normalization accumulator
            bucket_list = []  # bucket names for concatenation
            num_datasets = 0  # number of enabled dataset configs
            datasets = {}  # bucket -> dataset instance
            sample_weights = []  # bucket sampling weights
            
            if self.driving_dataset is not None or self.dreamer_dataset is not None:
                # Create lists of datasets and their corresponding training partitions
                used_driving_datasets = []  # enabled dataset configs
                used_train_partitions = []  # matching bucket weight maps
                
                # Pair datasets with their training partitions and filter out None datasets
                dataset_pairs = zip(
                    [self.driving_dataset, self.dreamer_dataset],
                    [self.train_partitions, self.train_partitions_dreamer]
                )
                
                for dataset, partition in dataset_pairs:
                    if dataset is not None:
                        used_driving_datasets.append(dataset)  # keep config
                        used_train_partitions.append(partition)  # keep partitions
                
                weights_driving = 0.5  # split weight for driving
                weights_dreamer = 1 - weights_driving  # split weight for dreamer
                for udd_i, (used_driving_dataset, used_train_partitions) in enumerate(zip(used_driving_datasets, used_train_partitions)):
                    num_datasets += 1  # count enabled datasets
                    if used_train_partitions is not None:
                        bucket_list_tmp = list(used_train_partitions.keys())  # bucket names
                        sample_weights_tmp = list(used_train_partitions.values())  # bucket weights
                        sum_sample_weights = sum(sample_weights_tmp)  # normalize sum
                        sample_weights_tmp = [w / sum_sample_weights for w in sample_weights_tmp]  # normalize
                        if self.driving_dataset is not None and self.dreamer_dataset is not None:
                            if udd_i == 0:
                                sample_weights_tmp = [w * weights_driving for w in sample_weights_tmp]  # scale driving
                            else:
                                sample_weights_tmp = [w * weights_dreamer for w in sample_weights_tmp]  # scale dreamer
                        if udd_i == 1:
                            bucket_list_tmp = [f"{b}_dreamer" for b in bucket_list_tmp]  # disambiguate dreamer buckets
                        bucket_list.extend(bucket_list_tmp)  # add bucket names
                        sample_weights.extend(sample_weights_tmp)  # add weights
                    else:
                        if udd_i == 1:
                            bucket_list_tmp = ['all_dreamer']  # fallback dreamer bucket
                            sample_weights_tmp = [weights_dreamer]  # fallback dreamer weight
                        else:
                            bucket_list_tmp = ['all']  # fallback driving bucket
                            sample_weights_tmp = [weights_driving]  # fallback driving weight
                        bucket_list.extend(bucket_list_tmp)  # add bucket names
                        sample_weights.extend(sample_weights_tmp)  # add weights
                    

                    for bucket in bucket_list_tmp:
                        bucket_name = bucket.replace('_dreamer', '')  # strip suffix for dataset
                        datasets[bucket] = hydra.utils.instantiate(
                            used_driving_dataset,
                            split="train",
                            bucket_name=bucket_name,
                            **self.cfg,
                            **self.base_dataset,
                            _recursive_=False
                        )
                    self.val_datasets.append(hydra.utils.instantiate(
                            used_driving_dataset,
                            split="val",
                            bucket_name="all",
                            **self.cfg,
                            **self.base_dataset,
                            _recursive_=False
                        ))
                    sum_sample_weights = sum(sample_weights_tmp)  # track last sum
            

            
            self.train_dataset = None  # default to None if no data
            if len(datasets) > 0:
                # remove datasets with 0 samples
                sample_weights = [sample_weights[i] for i, bucket in enumerate(bucket_list) if datasets[bucket].__len__() > 0]
                bucket_list = [bucket for bucket in bucket_list if datasets[bucket].__len__() > 0]
                if len(bucket_list) != len(datasets):
                    # print in red
                    print(f"\033[91mDatasets with 0 samples: {set(datasets.keys()) - set(bucket_list)}\033[00m")
                    print(f"\033[91mContinue without this bucket.\033[00m")
                datasets = {key: value for key, value in datasets.items() if value.__len__() > 0}
                # concatenate datasets across buckets
                self.train_dataset = torch.utils.data.ConcatDataset([datasets[bucket] for bucket in bucket_list])
                # expand bucket weights into per-sample weights
                weights_train = [[sample_weights[i]] * datasets[bucket].__len__() for i, bucket in enumerate(bucket_list)]
                weights_train = list(itertools.chain.from_iterable(weights_train))
                # normalize sampling length to smallest bucket budget
                num_samples_all = [datasets[bucket].__len__() // sample_weights[i] for i, bucket in enumerate(bucket_list)]
                num_samples = int(min(num_samples_all))  # * num_datasets
                print(f"Num samples: {num_samples}")
                if self.driving_dataset is not None:
                    print(f"Num samples all: {datasets['all'].__len__()}")
                # sampler enforces weighted bucket sampling
                self.sampler_train = torch.utils.data.WeightedRandomSampler(weights=weights_train, num_samples=num_samples, replacement=True)
            # concatenate validation datasets
            self.val_dataset = torch.utils.data.ConcatDataset(self.val_datasets)
            self.predict_dataset = None  # not used in train/val mode

        else:
            if self.qa_dataset is not None:
                predict_dataset = self.qa_dataset  # QA/commentary eval dataset
            elif self.insteval_dataset is not None:
                predict_dataset = self.insteval_dataset  # dreamer eval dataset
            # build the evaluation dataset
            self.predict_dataset = hydra.utils.instantiate(
                    predict_dataset,
                    split="val",
                    bucket_name="all",
                    **self.cfg,
                    **self.base_dataset,
                    _recursive_=False
                )


    def train_dataloader(self):
        """Create the training DataLoader or return None if empty."""
        if self.train_dataset is None:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            # shuffle=True, # we use custom sampler instead
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
            sampler=self.sampler_train,
            pin_memory=True,
        )

    def predict_dataloader(self):
        """Create the prediction DataLoader for evaluation runs."""
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
            pin_memory=True,
        )


    @line_profiler.profile
    def dl_collate_fn(self, data):
        """Collate a list of DatasetOutput samples into a DrivingExample.
        #the the input is of type dataset output.

        This prepares image tensors, tokenized prompts, waypoint labels,
        and auxiliary evaluation metadata used by the model.
        """
        # Batch size and per-image patch grid settings.
        BS = len(data)
        grid_nums = [self.NUM_IMAGE_PATCHES] # we split the front forward into two patches (1x2)

        # ------------------------------------------------------------------
        # Drop any sample whose number of frames (T) doesn't match the batch
        # majority. This guards against rare cases where the index-time ICL
        # filter misses something (e.g. stale icl_examples.json.gz, image
        # missing on disk so load_icl_image returned None at runtime).
        # We log the offending sample's measurement_path so it can be fixed.
        # ------------------------------------------------------------------
        _T_per_sample = []
        for i in range(BS):
            _icl_imgs = data[i].icl_images
            _T_per_sample.append((len(_icl_imgs) + 1) if _icl_imgs else 1)
        if len(set(_T_per_sample)) > 1:
            from collections import Counter
            _majority_T = Counter(_T_per_sample).most_common(1)[0][0]
            _kept, _dropped = [], []
            for i, t in enumerate(_T_per_sample):
                (_kept if t == _majority_T else _dropped).append(i)
            print(f"\033[93m[collate] Inconsistent T in batch ({_T_per_sample}). "
                  f"Keeping majority T={_majority_T}, dropping {len(_dropped)} sample(s):\033[0m")
            for i in _dropped:
                print(f"\033[93m  -> T={_T_per_sample[i]}  measurement_path={data[i].measurement_path}\033[0m")
            data = [data[i] for i in _kept]
            BS = len(data)
            if BS == 0:
                # Pathological case: every sample mismatched. Re-raise so the
                # DataLoader can move on to the next batch instead of silently
                # producing an empty batch that breaks downstream tensor ops.
                raise RuntimeError("[collate] All samples had inconsistent T; batch is empty")

        # Hold processed image tensors and their sizes.
        image_ff_pixel, image_ff_sizes = None, None
        # Store original front camera image sizes for each sample.
        image_ff_org = torch.tensor(np.asarray([data[i].image_ff_org_size for i in range(BS)]))
            
        # Iterate over configured camera streams (currently just front-forward).
        for idx, img_to_consider in enumerate(self.IMAGES_TO_CONSIDER):
            # Grab a sample image tensor to infer shape.
            img_tmp = getattr(data[0], img_to_consider)
            #1 ,3,359,1024
            _, C, H, W = img_tmp.shape

            # Build per-sample frame sequences: [icl_img_1, ..., query_img] -> [T, C, H, W].
            # T=1 when ICL is off, T=num_icl+1 when ICL examples are present.
            all_sample_images = []
            T_per_sample = []
            for i in range(BS):
                query_img = getattr(data[i], img_to_consider)
                if query_img is None:
                    query_img = np.zeros_like(img_tmp)
                icl_imgs = data[i].icl_images  # list of [1, C, H, W] or None
                if icl_imgs:
                    frames = np.concatenate(icl_imgs + [query_img], axis=0)  # [T, C, H, W] #T-> num_icl+1
                else:
                    frames = query_img  # [1, C, H, W]
                all_sample_images.append(frames)
                T_per_sample.append(frames.shape[0])

            # All samples in the batch must have the same T. The pre-filter
            # above should guarantee this; the assertion here is belt-and-
            # suspenders. If it ever fires, surface the offending paths.
            T = T_per_sample[0]
            if not all(t == T for t in T_per_sample):
                debug_paths = [data[i].measurement_path for i in range(BS)]
                raise AssertionError(
                    f"Inconsistent number of images per sample in batch:\n"
                    f"  T_per_sample={T_per_sample}\n"
                    f"  measurement_paths={debug_paths}"
                )

            # Flatten to [BS*T, C, H, W] for batch preprocessing.
            images_batch_tensor = torch.tensor(np.asarray(all_sample_images)).float()
            images_batch_tensor = images_batch_tensor.view(BS * T, C, H, W)
            images_batch_list = list(images_batch_tensor)

            # Preprocess images with the InternVL2 pipeline.
            if 'internvl2' in self.encoder_variant.lower():
                images_processed = preprocess_image_batch(images_batch_list, input_size=448, use_global_img=self.use_global_img, max_num_grid=grid_nums[idx])
            else:
                raise ValueError(f"Image preprocessing for {self.encoder_variant} not implemented")

            # Extract processed pixels and size metadata.
            images_pixel = images_processed['pixel_values']
            image_sizes = images_processed['image_sizes']

            # Reassemble into [B, T, NP, C, H', W'].
            assert images_pixel.shape[0] == BS * T
            num_patches = images_pixel.shape[1]
            new_height = images_pixel.shape[3]
            new_width = images_pixel.shape[4]
            images_pixel = images_pixel.view(BS, T, num_patches, C, new_height, new_width)

            # Store front camera outputs (other image keys are not supported).
            if img_to_consider == 'image_ff':
                image_ff_pixel = images_pixel
                image_ff_sizes = image_sizes
            else:
                raise ValueError(f"Image type {img_to_consider} not supported")

        # Apply the InternVL2 chat template to each conversation:
        #   - conversation_dict: full prompt (user + assistant) used during training
        #   - question_dict:     user-only prompt (no assistant answer) used during inference
        # Each dict contains tokenized ids, valid/mask tensors, loss mask, and raw strings.
        # <image> tokens in the text are replaced here with <img><IMG_CONTEXT>xN</img> blocks,
        # where N = num_image_tokens_total (patches x tokens_per_patch, e.g. 2x256=512).
        # With ICL enabled each conversation has multiple <image> tokens (one per ICL example
        # + one for the query), all expanded in order to match the images sent to the encoder.
        conversations = [data[i].conversation for i in range(BS)]
        conversation_dict, question_dict = get_custom_chat_template(conversations, self.tokenizer, self.encoder_variant, self.num_image_tokens_total)

        # Build the placeholder map for each sample:
        # Some text tokens are special markers for continuous values (e.g. <TARGET_POINT>).
        # Here we convert the string token key -> integer token ID, so the model's adaptor
        # can find and replace those embedding slots with learned continuous embeddings
        # (e.g. target waypoint coordinates encoded by WaypointInputAdaptor).
        placeholder_batch_list = []
        for i in range(BS):
            tmp = {}
            for key, value in data[i].placeholder_values.items():
                token_nr_key = self.tokenizer.convert_tokens_to_ids(key)
                tmp[token_nr_key] = value
            placeholder_batch_list.append(tmp)

        # Used during TRAINING: contains full conversation (user + assistant).
        # The model sees the full sequence but only computes loss on assistant tokens (loss_masking).
        prompt_languagelabel = LanguageLabel(
            phrase_ids=conversation_dict['phrase_ids'],         # [B, max_len] int64 - token IDs (includes padding, image tokens, text tokens)
            phrase_valid=conversation_dict['phrase_valid'],     # [B, max_len] bool  - True = real token, False = padding (don't feed to model)
            phrase_mask=conversation_dict['phrase_mask'],       # [B, max_len] bool  - same as phrase_valid, reserved for future attention masking
            placeholder_values=placeholder_batch_list,         # list of {token_id: value} - continuous values to replace special tokens (e.g. <TARGET_POINT> -> waypoint coords)
            language_string=conversation_dict['language_string'],  # list of raw strings - human-readable prompt, used for logging/debug only
            loss_masking=conversation_dict['loss_masking'],    # [B, max_len] bool  - True ONLY at assistant answer positions, loss is zero everywhere else
        )

        # Used during INFERENCE: contains only the user prompt (no assistant answer).
        # The model generates the answer autoregressively starting from this prefix.
        # loss_masking is all False here since there is nothing to supervise.
        prompt_question_languagelabel = LanguageLabel(
            phrase_ids=question_dict['phrase_ids'],         # [B, max_len] int64 - token IDs of user prompt only
            phrase_valid=question_dict['phrase_valid'],     # [B, max_len] bool  - True = real token, False = padding
            phrase_mask=question_dict['phrase_mask'],       # [B, max_len] bool  - same as phrase_valid
            placeholder_values=placeholder_batch_list,     # same placeholder map as training prompt
            language_string=question_dict['language_string'],  # raw user prompt string (no assistant part)
            loss_masking=question_dict['loss_masking'],    # [B, max_len] bool  - all False (no answer tokens to supervise)
        )
        # Extract answer strings and wrap them as labels.
        answer_string_list = [data[i].answer[0]['content'][0]['text'] for i in range(BS)]
        answer_label =  LanguageLabel(
            phrase_ids=None,
            phrase_valid=None,
            phrase_mask=None,
            placeholder_values=None,
            language_string=answer_string_list,
            loss_masking=None,
        )
        
        # Select 1D or 2D waypoint targets based on config.
        if self.base_dataset.use_1d_wps:
            waypoints = torch.tensor(np.asarray([data[i].waypoints_1d for i in range(len(data))])).float() # [B, F, 2] 11 future waypoints 0.2s apart
        else:
            waypoints = torch.tensor(np.asarray([data[i].waypoints for i in range(len(data))])).float() # [B, F, 2] 11 future waypoints 0.2s apart
        
        # Keep eval-only metadata when running in predict mode.
        if self.predict:
            qa_templates = [data[i].qa_templates[0] if data[i].qa_templates is not None else None for i in range(BS) ]
            eval_infos = [data[i].eval_infos if data[i].eval_infos is not None else None for i in range(BS) ]
        else:
            qa_templates = None
            eval_infos = None
        
        # Assemble the model input bundle.
        driving_input=DrivingInput(
                camera_images=image_ff_pixel,  # [B, T, N, C, H, W] uint8 [0, 255]
                image_sizes=image_ff_sizes,
                camera_intrinsics = torch.repeat_interleave(get_camera_intrinsics(W, H, 110).unsqueeze(0), BS, dim=0).view(BS, 3, 3).float(),
                camera_extrinsics = torch.repeat_interleave(get_camera_extrinsics().unsqueeze(0), BS, dim=0).view(BS, 4, 4).float(),
                vehicle_speed=torch.tensor(np.asarray([data[i].speed for i in range(len(data))])).float(),  # [B, S] float32
                target_point=torch.tensor(np.asarray([data[i].target_points for i in range(len(data))])).float(),  # [B, 2] float32
                prompt=prompt_languagelabel,
                prompt_inference=prompt_question_languagelabel,
            )

        # Assemble the supervision labels.
        driving_label=DrivingLabel(
                waypoints=waypoints,
                path=torch.tensor(np.asarray([data[i].path for i in range(len(data))])).float(), # [B, 3, RH, RW] uint8 [0, 255]
                answer=answer_label,
                image_ff_org=image_ff_org,
                eval_infos=eval_infos,
            )
            
        # Return a batch object that combines inputs, labels, and metadata.
        return DrivingExample(
            driving_input=driving_input,
            driving_label=driving_label,
            run_id=encode_uint8([data[i].measurement_path for i in range(BS)], 1000),  # [B] str
            qa_templates=qa_templates,
        )

    def dl_collate_fn_val(self, data):
        """Placeholder for a validation-specific collate function."""
        pass

    def dl_collate_fn_test(self, data):
        """Placeholder for a test-specific collate function."""
        pass


@hydra.main(config_path=f"../config", config_name="config", version_base="1.1")
def test(cfg):
    """Quick dataset sanity check for local debugging."""
    
    get_waypoint_stats = True
    
        
    processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True, revision=hf_revision(cfg.model.vision_model.variant))
    dm = hydra.utils.instantiate(
        cfg.data_module,
        processor=processor,
        # tokenizer=llm_tokenizer,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant="llava-hf/llava-v1.6-mistral-7b-hf",
        _recursive_=False
    )


    dm.setup()
    dl = dm.val_dataloader()
    print(dl.dataset.__len__())

    iterations = 0
    
    all_waypoints = []
    all_waypoints_diff = []
    for batch in dl:
        
        iterations += 1
        
        if iterations % 100 == 0:
            print(f"Iteration: {iterations}")
        
        if iterations > 20000:
            break
        
        if get_waypoint_stats:
            # get stats about range of waypoints
            waypoints = batch.driving_label.waypoints
            all_waypoints.append(waypoints)
            
            # get residuals
            residuals = waypoints[:,1:] - waypoints[:,:-1]
            all_waypoints_diff.append(residuals)
            
    # get histogram of waypoints
    if get_waypoint_stats:
        all_waypoints = torch.cat(all_waypoints, dim=0)
        all_waypoints_diff = torch.cat(all_waypoints_diff, dim=0)
        
        all_waypoints = all_waypoints.view(-1, 2)
        all_waypoints_diff = all_waypoints_diff.view(-1, 2)
        
        
        
        import matplotlib.pyplot as plt
        plt.hist(all_waypoints[:,0].numpy(), bins=100)
        plt.savefig('waypoints_x.png')
        max_x = all_waypoints[:,0].max().item()
        min_x = all_waypoints[:,0].min().item()
        print(f"Max x: {max_x}, Min x: {min_x}")
        plt.clf()
        plt.hist(all_waypoints[:,1].numpy(), bins=100)
        plt.savefig('waypoints_y.png')
        max_y = all_waypoints[:,1].max().item()
        min_y = all_waypoints[:,1].min().item()
        print(f"Max y: {max_y}, Min y: {min_y}")
        plt.clf()
        
        plt.hist(all_waypoints_diff[:,0].numpy(), bins=100)
        plt.savefig('waypoints_diff_x.png')
        max_x_diff = all_waypoints_diff[:,0].max().item()
        min_x_diff = all_waypoints_diff[:,0].min().item()
        print(f"Max x diff: {max_x_diff}, Min x diff: {min_x_diff}")
        plt.clf()
        plt.hist(all_waypoints_diff[:,1].numpy(), bins=100)
        plt.savefig('waypoints_diff_y.png')
        max_y_diff = all_waypoints_diff[:,1].max().item()
        min_y_diff = all_waypoints_diff[:,1].min().item()
        print(f"Max y diff: {max_y_diff}, Min y diff: {min_y_diff}")
        plt.clf()
            
            

if __name__ == "__main__":
    test()