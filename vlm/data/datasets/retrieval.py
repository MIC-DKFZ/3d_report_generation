"""Retrieved report and image context for CT-RATE, Merlin, and Bonn."""

import json
from pathlib import Path
from typing import Dict, List, Literal, Tuple

from vlm.configs.run_config import RunConfig

from .base import SYSTEM_TEXT
from .bonn import BonnDataset
from .ctrate import CTRATEDataset
from .merlin import MerlinDataset


class CTRATEDatasetRetrieval(CTRATEDataset):
    def __init__(
        self,
        cfg: RunConfig,
        vision_encoder_patch_size: Tuple[int, ...],
        split: Literal["train", "val", "test"] = "train",
        n_samples: int | None = None,
    ):
        super().__init__(
            cfg=cfg,
            vision_encoder_patch_size=vision_encoder_patch_size,
            split=split,
            n_samples=n_samples,
        )
        # FIXME: not sure how to handle the case where the training samples are limited by `n_train_samples`
        # E.g. a check would be needed that the image IDs of thre retrieval json are all included in the training subset
        self.retrieval_json_path = Path(cfg.data.retrieval_json_path)
        # TODO: provide this part of the cfg
        self.n_retrieval_samples = int(cfg.data.retrieval_n_samples)
        self.maybe_add_vision_tokens = ""
        if self.split in ["val", "test"]:
            self.dataset_json_path_for_retrieval = Path(cfg.data.train_dataset_json_path)
            with open(self.dataset_json_path_for_retrieval, "r") as f:
                self.samples_retrieval = json.load(f)
        elif self.split == "train":
            self.samples_retrieval = self.samples

        self.additional_instruction_prompt = "\nConsider those reports which were retrieved based on the query image."
        self.add_image = False

        with open(self.retrieval_json_path, "r") as f:
            self.retrieval_dict = self._sanatize_retrieval_dict(json.load(f), self.samples_retrieval)

        self._validate_retrieval_dict_covers_samples(self.samples, self.retrieval_dict)

    @staticmethod
    def _sanatize_retrieval_dict(
            retrieval_dict: Dict[str, Dict],
            samples_retrieval: List[Dict]
        ) -> Dict[str, Dict]:
        # 1. Create a Set of valid IDs for O(1) lookup speed
        valid_ids = {sample['image_id'] for sample in samples_retrieval}

        for _, data in retrieval_dict.items():
            neighbors = data.get('k_nearest_neighbor', [])
            scores = data.get('similarity', [])
            assert len(neighbors) == len(scores)

            clean_neighbors = []
            clean_scores = []

            for neighbor_id, score in zip(neighbors, scores):
                if neighbor_id in valid_ids:
                    clean_neighbors.append(neighbor_id)
                    clean_scores.append(score)
            data['k_nearest_neighbor'] = clean_neighbors
            data['similarity'] = clean_scores

        return retrieval_dict

    @staticmethod
    def _validate_retrieval_dict_covers_samples(
            samples: List[Dict],
            retrieval_dict: Dict[str, Dict],
        ) -> None:
        missing_image_ids = []

        for sample in samples:
            if "image_id" not in sample:
                raise KeyError(
                    "Encountered a training sample without an `image_id` key while validating retrieval coverage."
                )

            image_id = sample["image_id"]
            if image_id not in retrieval_dict:
                missing_image_ids.append(image_id)

        if missing_image_ids:
            n_missing = len(missing_image_ids)
            n_samples = len(samples)
            preview_missing = missing_image_ids[:10]
            raise KeyError(
                "Retrieval dictionary is missing entries for training samples. "
                f"Missing {n_missing}/{n_samples} image_ids. "
                f"First missing ids: {preview_missing}"
            )

    @staticmethod
    def get_case_from_dataset_json(dataset_json: List[Dict], image_id: str) -> Dict[str, str]:
        for case in dataset_json:
            if case["image_id"] == image_id:
                return case

        raise KeyError(
            f"Image id: {image_id} can not be found in provided dataset_json make sure it exists"
        )

    def __getitem__(self, index):
        result = super().__getitem__(index)
        result["formatted_qa"], result["formatted_qa_prompt_only"] = self.modify_input(
            result["formatted_qa"], result["formatted_qa_prompt_only"], result
        )
        return result

    def modify_input(self, formatted_qa, formatted_qa_prompt_only, result):
        if self.n_retrieval_samples == 0:
            return formatted_qa, formatted_qa_prompt_only

        query_img_id = result["image_id"]

        # get image_ids of most similar
        nearest_neighbors_ids = self.retrieval_dict[query_img_id]["k_nearest_neighbor"]
        nearest_neighbors_ids = nearest_neighbors_ids[:min(len(nearest_neighbors_ids), self.n_retrieval_samples)]

        # formatted_qa and formatted_qa_prompt_only reference the same "user_message" object
        # we only need to modify one of them here
        assert formatted_qa[1] is formatted_qa_prompt_only[1]
        # for input_prompt in [formatted_qa, formatted_qa_prompt_only]:
        # get user input
        for d in formatted_qa:
            if d["role"] == "user":
                d["content"].append({"type": "text", "text": self.additional_instruction_prompt})
                for n_nn, nearest_neighbor_id in enumerate(nearest_neighbors_ids):
                    case = self.get_case_from_dataset_json(self.samples_retrieval, nearest_neighbor_id)
                    answer = case["answer"]
                    if self.add_image:
                        d["content"].append(
                            {"type": "vision", "vision_input_path": nearest_neighbor_id}
                        )
                    d["content"].append({"type": "text", "text": f"Report {n_nn + 1}:\n{answer}"})

        return formatted_qa, formatted_qa_prompt_only


class CTRATEDatasetRetrievalWithImage(CTRATEDatasetRetrieval):
    def __init__(
        self,
        cfg: RunConfig,
        vision_encoder_patch_size: Tuple[int, ...],
        split: Literal["train", "val", "test"] = "train",
        n_samples: int | None = None,
    ):
        super().__init__(
            cfg=cfg,
            vision_encoder_patch_size=vision_encoder_patch_size,
            split=split,
            n_samples=n_samples,
        )
        # self.maybe_add_vision_tokens = "<|vision_start|><|video_pad|><|vision_end|>"
        self.maybe_add_vision_tokens = ""
        self.additional_instruction_prompt = "\nConsider those similar report/image pairs which were retrieved based on the query image."
        self.add_image = True


    # TODO: instead of overriding the signature of the base class refactor base class and introduce hooks where this can be used
    def _get_vision_input_path(self, image_id: str) -> List[Path]:
        # get image_ids of most similar
        nearest_neighbors_ids = self.retrieval_dict[image_id]["k_nearest_neighbor"]
        nearest_neighbors_ids = nearest_neighbors_ids[:min(len(nearest_neighbors_ids), self.n_retrieval_samples)]

        # Bind the super method to a variable outside the list comprehension as else might lead to TypeError in some python versions
        parent_get_vision_input_path = super()._get_vision_input_path

        # TODO: the fix with split path is VERY hacky and not sure if it is correct overall...
        paths = [parent_get_vision_input_path(nn_id, split_path="train") for nn_id in nearest_neighbors_ids]

        paths.insert(0, parent_get_vision_input_path(image_id))  # add query image at first position
        return paths

    def _load_vision_input(self, vision_input_path: List[Path]):
        parent_load_vision_input = super()._load_vision_input
        res = [parent_load_vision_input(p) for p in vision_input_path]
        # FIXME: this function assumes that each input has the same size...
        # assert [len(r.shape) == 5 for r in res], "Each loaded vision input should be 5D (B, C, H, W, D)"
        # return np.concatenate(res, axis=0)  # concatenate along batch dimension
        # Flatten again
        # res = [image_in_case for same_case in res for image_in_case in same_case]
        return res

    def _load_embeddings(self, sample, result: Dict) -> Dict:
        vision_input_path = self._get_vision_input_path(sample["image_id"])
        vision_input = self._load_vision_input(vision_input_path)

        if self.embedding_transforms is not None:
            vision_input = [self.embedding_transforms(embedding) for embedding in vision_input]
        result["vision_input"] = vision_input

        return result


class CTRATEDatasetRetrievalWithImagePrompt1(CTRATEDatasetRetrievalWithImage):
    def __init__(
        self,
        cfg: RunConfig,
        vision_encoder_patch_size: Tuple[int, ...],
        split: Literal["train", "val", "test"] = "train",
        n_samples: int | None = None,
    ):
        super().__init__(
            cfg=cfg,
            vision_encoder_patch_size=vision_encoder_patch_size,
            split=split,
            n_samples=n_samples,
        )
        # self.maybe_add_vision_tokens = "<|vision_start|><|video_pad|><|vision_end|>"
        self.maybe_add_vision_tokens = ""
        self.additional_instruction_prompt = "\nRetrieved Similar Cases:"
        self.add_image = True

    def format_into_message(
            self, question, answer, system_text=SYSTEM_TEXT
    ):

        assert not self.text_only
        system_text = "You are a medical imaging AI assistant specialized in analyzing CT scans. Your task is to generate a structured radiology report based on the provided CT scan, using retrieved similar cases as contextual guidance."
        system_message = {"role": "system", "content": [{"type": "text", "text": system_text}]}

        question_p1 = """
Carefully analyze the query image and consider the retrieved image-report pairs. Use the retrieved reports to:
- Understand common language and structure for describing findings.
- Identify potential pathologies that may appear in similar scans.
- Avoid contradicting visual evidence in the query image.

However, do **not** copy findings verbatim. Always base your analysis on the **query image**. If the retrieved cases differ significantly, prioritize what you observe.

Output a report in the following format:

**Findings:**
- Describe observed abnormalities (e.g., location, size, density, morphology).
- Mention normal findings when relevant for context.
- Use precise anatomical terminology.

**Impression:**
- Summarize the most significant findings.
- Suggest possible diagnoses or differential considerations if supported by imaging.
- Avoid overinterpretation or speculation beyond visual evidence.

Now analyze the following:
        """

        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": question_p1}, {"type": "vision"}],
        }

        answer_message = {"role": "assistant", "content": [{"type": "text", "text": answer}]}

        full_conversation = [system_message, user_message, answer_message]
        prompt_only_conversation = [system_message, user_message]
        return full_conversation, prompt_only_conversation

    def modify_input(self, formatted_qa, formatted_qa_prompt_only, result):
        if self.n_retrieval_samples == 0:
            return formatted_qa, formatted_qa_prompt_only

        query_img_id = result["image_id"]

        # get image_ids of most similar
        nearest_neighbors_ids = self.retrieval_dict[query_img_id]["k_nearest_neighbor"]
        nearest_neighbors_ids = nearest_neighbors_ids[:min(len(nearest_neighbors_ids), self.n_retrieval_samples)]

        # formatted_qa and formatted_qa_prompt_only reference the same "user_message" object
        # we only need to modify one of them here
        assert formatted_qa[1] is formatted_qa_prompt_only[1]
        # for input_prompt in [formatted_qa, formatted_qa_prompt_only]:
        # get user input
        for d in formatted_qa:
            if d["role"] == "user":
                d["content"].append({"type": "text", "text": self.additional_instruction_prompt})
                for n_nn, nearest_neighbor_id in enumerate(nearest_neighbors_ids):
                    case = self.get_case_from_dataset_json(self.samples_retrieval, nearest_neighbor_id)
                    answer = case["answer"]
                    d["content"].append({"type": "text", "text": f"[Case {n_nn + 1}]\n"})
                    if self.add_image:
                        d["content"].append(
                            {"type": "vision", "vision_input_path": nearest_neighbor_id}
                        )
                    d["content"].append({"type": "text", "text": f"Report:\n{answer}"})

        return formatted_qa, formatted_qa_prompt_only


class MerlinDatasetRetrieval(MerlinDataset, CTRATEDatasetRetrieval):
    pass


class MerlinDatasetRetrievalWithImage(MerlinDataset, CTRATEDatasetRetrievalWithImage):
    # Use Merlin path conventions for both query and retrieved neighbor images.
    def _get_vision_input_path(self, image_id: str, split_path=None) -> List[Path]:
        nearest_neighbors_ids = self.retrieval_dict[image_id]["k_nearest_neighbor"]
        nearest_neighbors_ids = nearest_neighbors_ids[
            : min(len(nearest_neighbors_ids), self.n_retrieval_samples)
        ]

        paths = [MerlinDataset._get_vision_input_path(self, nn_id) for nn_id in nearest_neighbors_ids]
        paths.insert(0, MerlinDataset._get_vision_input_path(self, image_id))
        return paths


class BonnDatasetRetrieval(BonnDataset, CTRATEDatasetRetrieval):
    pass


class BonnDatasetRetrievalWithImage(BonnDataset, CTRATEDatasetRetrievalWithImage):
    pass
