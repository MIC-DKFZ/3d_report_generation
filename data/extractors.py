from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict


class RequiredKeys(TypedDict):
    """
    Defines the required keys for the dictionary.
    """

    image_id: str
    question: str
    answer: str
    answer_type: str
    system_text: str


class QAItemDict(RequiredKeys):
    """
    A dictionary-like class that inherits from TypedDict and enforces the presence of required keys.
    """

    def __init__(self, data: Dict[str, Any]):
        """
        Initializes the MyDict object.

        Args:
            data: A dictionary containing the initial data.

        Raises:
            ValueError: If any of the required keys are missing.
        """
        # Validate required keys
        required_keys = set(QAItemDict.__required_keys__)  # Access required keys from TypedDict
        if not required_keys.issubset(data.keys()):
            missing_keys = required_keys - data.keys()
            raise ValueError(f"Missing required keys: {missing_keys}")

        super().__init__(**data)  # Initialize TypedDict with the data

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Allows setting items using the square bracket notation (e.g., my_dict["key1"] = "new value").
        """
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        """
        Allows deleting items using the del statement (e.g., del my_dict["key1"]).
        """
        super().__delitem__(key)

    @property
    def __required_keys__(self) -> Set[str]:
        """
        Returns the set of required keys for the TypedDict.
        """
        return set(self.__annotations__.keys())


class AMOSMMExtractor:
    """
    Extractor for AMOS-MM dataset that separates data extraction from formatting.
    """

    @staticmethod
    def extract_qa(qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all information from an AMOS-MM QA pair.
        """
        # Extract basic data
        options_text = "\n".join([f"{key}. {value}" for key, value in qa_pair["options"].items()])
        correct_option_text = qa_pair["options"][qa_pair["answer"]]
        reasoning = qa_pair.get("reasoning", "")

        # Generate formatted texts
        user_text = f"{qa_pair['question']}\n\n{options_text}"
        full_answer = f"The answer is {qa_pair['answer']}. {correct_option_text}. {reasoning}"

        return {
            "qa_question": qa_pair["question"],
            "qa_options": qa_pair["options"],
            "answer": qa_pair["answer"],
            "answer_type": "CHOICE",
            "answer_subtype": "MULTIPLE_CHOICE",
            "reasoning": reasoning,
            "correct_option_text": correct_option_text,
            "options_text": options_text,
            "content_type": qa_pair["type"],
            "question": user_text,
            "user_text": user_text,
            "full_answer": full_answer,
            "system_text": "You are a medical imaging AI trained to answer questions about CT scans.",
        }

    @staticmethod
    def extract_finding_question(region: str, findings: str) -> Dict[str, Any]:
        """
        Extract formatted texts for region questions.
        """
        if not findings:
            return None

        return {
            "region": region,
            "findings": findings,
            "answer_type": "OPEN",
            "answer_subtype": "FINDINGS",
            "user_text": f"How would you describe the {region} region of this CT scan?",
            "question": f"How would you describe the {region} region of this CT scan?",
            "full_answer": findings,
            "answer": findings,
            "system_text": "You are a medical imaging AI trained to analyze CT scans.",
        }

    @staticmethod
    def extract_impressions_question(region: str, impressions: str) -> Dict[str, Any]:
        """
        Extract formatted texts for impression questions.
        """
        if not impressions:
            return None

        if isinstance(impressions, list):
            impressions = ", ".join(impressions)

        return {
            "region": region,
            "impressions": impressions,
            "answer_type": "OPEN",
            "answer_subtype": "IMPRESSIONS",
            "user_text": f"What can you detect in the {region} region of this CT scan?",
            "question": f"What can you detect in the {region} region of this CT scan?",
            "full_answer": impressions,
            "answer": impressions,
            "system_text": "You are a medical imaging AI trained to analyze CT scans.",
        }

    @staticmethod
    def extract_report(report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all information from an AMOS-MM report.
        """
        extracted_report = {
            "raw": report,  # Keep the original data
            "finding_questions": {},
            "impression_question": {},
            "impression": report.get("impression", {}),
            "findings": report.get("findings", {}),
        }

        # Extract region findings and create region questions
        if "findings" in report:
            for region in ["chest", "abdomen", "pelvis"]:
                region_findings = report["findings"].get(region, "")

                # Create formatted region question
                if region_findings:
                    extracted_report["finding_questions"][region] = AMOSMMExtractor.extract_finding_question(
                        region, region_findings
                    )

        # Extract impression and create impression question
        if "impression" in report:
            for region in ["chest", "abdomen", "pelvis"]:
                region_impressions = report["impression"].get(region, "")

                # Create formatted impression question
                if region_impressions:
                    extracted_report["impression_question"][region] = AMOSMMExtractor.extract_impressions_question(
                        region, region_impressions
                    )

        return extracted_report

    @staticmethod
    def extract_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all relevant information from an AMOS-MM sample.
        """
        extracted = {
            "image_path": sample["image"],
            "image_id": sample["image"],
            "qa_items": [],
            "report": {},
            "questions": [],
        }

        if "labels" in sample:
            if "report" in sample["labels"]:
                report_dict = AMOSMMExtractor.extract_report(sample["labels"]["report"])
                extracted["report"]["raw"] = report_dict["raw"]
                extracted["report"]["findings"] = report_dict["findings"]
                extracted["report"]["impression"] = report_dict["impression"]

                # Add finding questions
                for region, finding_questions in report_dict["finding_questions"].items():
                    extracted["qa_items"].append(finding_questions)

                # Add impression question
                for region, impression_question in report_dict["impression_question"].items():
                    extracted["qa_items"].append(impression_question)

            if "qa" in sample["labels"]:
                extracted["questions"] = sample["labels"]["qa"]

                for qa_pair in sample["labels"]["qa"]:
                    extracted["qa_items"].append(AMOSMMExtractor.extract_qa(qa_pair))

        return extracted

    @staticmethod
    def extract_sample_list(samples: List[Dict[str, Any]]) -> List[QAItemDict[str, Any]]:
        """
        Extract all relevant information from a list of AMOS-MM samples.
        """
        extracted_samples = []

        for sample in samples:
            extracted_sample = AMOSMMExtractor.extract_sample(sample)
            extracted_samples.append(extracted_sample)

        samples_flattened = []
        for sample in extracted_samples:
            cleaned_sample = {k: v for k, v in sample.items() if k != "qa_items"}
            if "qa_items" in sample:
                for qa_item in sample["qa_items"]:
                    flattened_sample = QAItemDict(
                        **cleaned_sample,
                        **qa_item,
                    )
                    samples_flattened.append(flattened_sample)

        return samples_flattened


class LIDCExtractor:
    """
    Extractor for LIDC dataset that separates data extraction from formatting.
    """

    @staticmethod
    def extract_qa(qa_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all information from a LIDC QA pair.
        """

        answer_subtype = "TEXT"  # Default to text answers

        if qa_item["answer_type"] == "CLOSED":
            answer_subtype = "YESNO"
        elif qa_item["answer_type"] == "OPEN":
            if isinstance(qa_item["answer"], int):
                answer_subtype = "NUMERIC"
            else:
                answer_subtype = "TEXT"

        # Basic information from the sample
        extracted = {
            "question": qa_item["question"],
            "answer": qa_item["answer"],
            "full_answer": qa_item["answer"],
            "answer_type": qa_item["answer_type"],
            "answer_subtype": answer_subtype,
        }

        # Generate formatted texts
        user_text = qa_item["question"]

        # Format the answer based on type
        if isinstance(qa_item["answer"], int):
            # For numerical answers (like count questions)
            full_answer = str(qa_item["answer"])
        else:
            # For text answers
            full_answer = qa_item["answer"]

        # Add formatted texts
        extracted.update(
            {
                "user_text": user_text,
                "full_answer": full_answer,
                "system_text": "You are a medical imaging AI trained to analyze lung CT scans and identify nodule characteristics.",
            }
        )

        return extracted

    @staticmethod
    def extract_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all relevant information from a LIDC sample.
        Since LIDC samples are already QA pairs, this mostly wraps extract_qa.
        """
        extracted_qa = LIDCExtractor.extract_qa(sample)

        # Structure it similar to the AMOS extractor's output format
        return {
            "series_uid": sample["series_uid"],
            "subject_id": sample["subject_id"],
            "scan_id": sample["scan_id"],
            "image_id": sample["series_uid"],
            "study_uid": sample["study_uid"],
            "content_type": sample["content_type"],
            "qid": sample["qid"],
            "item_qa": extracted_qa,
        }

    @staticmethod
    def extract_sample_list(samples: List[Dict[str, Any]]) -> List[QAItemDict[str, Any]]:
        """
        Extract all relevant information from a list of LIDC samples.
        """
        extracted_samples = []

        for sample in samples:
            extracted_sample = LIDCExtractor.extract_sample(sample)
            # extracted_samples.append(extracted_sample)

            cleaned_sample = {k: v for k, v in extracted_sample.items() if k != "item_qa"}
            if "item_qa" in extracted_sample:
                flattened_sample = QAItemDict(
                    **cleaned_sample,
                    **extracted_sample["item_qa"],
                )
                extracted_samples.append(flattened_sample)

        return extracted_samples
