import random
import re
from typing import Any, Dict, List, Type, TypedDict, TypeVar

ExtractorType = TypeVar("ExtractorType", bound="AMOSMMExtractor")


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
        required_keys = set(RequiredKeys.__annotations__.keys())
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")

        super().__init__(**data)  # Initialize TypedDict with the data


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
    def _extract_region_question(region: str, text: str, question_type: str) -> Dict[str, Any] | None:
        """
        Helper method to extract formatted texts for region questions.
        """
        if not text:
            return None

        if isinstance(text, list):
            text = ", ".join(text)

        question_prefix = "What are the findings in" if question_type == "FINDINGS" else "What is the impression of "

        return {
            "region": region,
            f"{question_type.lower()}": text,
            "answer_type": "OPEN",
            "answer_subtype": question_type,
            "user_text": f"{question_prefix} the {region.replace('_', ' ')} region of this CT scan?",  # TODO add prompt variety
            "question": f"{question_prefix} the {region.replace('_', ' ')} region of this CT scan?",
            "full_answer": text,
            "answer": text,
            "system_text": "You are a medical imaging AI trained to analyze CT scans.",
        }

    @staticmethod
    def extract_finding_question(region: str, findings: str) -> Dict[str, Any] | None:
        """
        Extract formatted texts for region questions.
        """
        return AMOSMMExtractor._extract_region_question(region, findings, "FINDINGS")

    @staticmethod
    def extract_impressions_question(region: str, impressions: str) -> Dict[str, Any] | None:
        """
        Extract formatted texts for impression questions.
        """
        return AMOSMMExtractor._extract_region_question(region, impressions, "IMPRESSIONS")

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
            #"image_path": sample["image"],
            "image_id": sample["image"],
            "qa_items": [],
            "report": {},
            "questions": [],
        }

        if "labels" in sample:
            if "report" in sample["labels"]:
                report_dict = AMOSMMExtractor.extract_report(sample["labels"]["report"])  # TODO Throw out report here?
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

    @classmethod
    def extract_sample_list(cls: Type[ExtractorType], samples: List[Dict[str, Any]]) -> List[QAItemDict]:
        """
        Extract all relevant information from a list of AMOS-MM samples.
        """
        samples_flattened = []

        for sample in samples:
            extracted_sample = cls.extract_sample(sample)
            cleaned_sample = {k: v for k, v in extracted_sample.items() if k != "qa_items"}

            for qa_item in extracted_sample.get("qa_items", []):
                flattened_sample = QAItemDict({**cleaned_sample, **qa_item})
                samples_flattened.append(flattened_sample)

        return samples_flattened


class AMOSMMRegionExtractor(AMOSMMExtractor):
    """
    Extractor for AMOS-MM dataset that separates data extraction from formatting.
    """

    @staticmethod
    def extract_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all relevant information from an AMOS-MM sample.
        """
        extracted = {
            #"image_path": sample["image"],
            "image_id": sample["original_case"],
            "qa_items": [],
            "report": {},
        }

        if "labels" in sample:
            findings = sample["labels"]
            extracted["report"]["findings"] = findings

            extracted["qa_items"].append(AMOSMMExtractor.extract_finding_question(sample["region"], findings))

        return extracted


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

        answer = qa_item["answer"]

        if qa_item["answer_type"] == "CLOSED":
            answer_subtype = "YESNO"
        elif isinstance(answer, int):
            answer_subtype = "NUMERIC"

        # Basic information with formatted texts
        return {
            "question": qa_item["question"],
            "answer": answer,
            "full_answer": answer,  # TODO remove duplicate
            "full_answer": answer,  # TODO remove duplicate
            "answer_type": qa_item["answer_type"],
            "answer_subtype": answer_subtype,
            "user_text": qa_item["question"],
            "system_text": "You are a medical imaging AI trained to analyze lung CT scans and identify nodule characteristics.",
        }

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
    def extract_sample_list(samples: List[Dict[str, Any]]) -> List[QAItemDict]:
        """
        Extract all relevant information from a list of LIDC samples.
        """
        extracted_samples = []

        for sample in samples:
            extracted_sample = LIDCExtractor.extract_sample(sample)
            cleaned_sample = {k: v for k, v in extracted_sample.items() if k != "item_qa"}
            if "item_qa" in extracted_sample:
                flattened_sample = QAItemDict({**cleaned_sample, **extracted_sample["item_qa"]})
                extracted_samples.append(flattened_sample)

        return extracted_samples


class CTRATEExtractor:
    """
    Extractor for CT-RATE dataset

    Processes CT-RATE dataset metadata for classification, report generation,
    and diagnosis tasks.
    """

    CONDITIONS = [
        "Arterial wall calcification",
        "Cardiomegaly",
        "Pericardial effusion",
        "Coronary artery wall calcification",
        "Hiatal hernia",
        "Lymphadenopathy",
        "Emphysema",
        "Atelectasis",
        "Lung nodule",
        "Lung opacity",
        "Pulmonary fibrotic sequela",
        "Pleural effusion",
        "Mosaic attenuation pattern",
        "Peribronchial thickening",
        "Consolidation",
        "Bronchiectasis",
        "Interlobular septal thickening",
    ]

    def __init__(self, shuffle_sentences: bool = False):
        """
        Initialize the extractor
        
        Args:
            shuffle_sentences (bool): If True, randomly shuffle sentences in text answers
        """
        self.shuffle_sentences = shuffle_sentences

    def _shuffle_sentences(self, text: str, p: float = 0.5) -> str:
        """
        Randomly shuffle sentences in a text with given probability
        
        Args:
            text (str): Input text to shuffle
            p (float): Probability of shuffling (default 0.5)
            
        Returns:
            str: Text with potentially shuffled sentences
        """
        # Handle non-string inputs and missing values
        if text is None:
            return ""
        
        # Convert to string and handle NaN values
        text_str = str(text).strip()
        if text_str.lower() in ['nan', 'none', '']:
            return ""
        
        if not self.shuffle_sentences or not text_str:
            return text_str
        
        # Split text into sentences using regex to handle various punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text_str)
        
        # Only shuffle if there are multiple sentences
        if len(sentences) > 1:
            # Randomly decide whether to shuffle based on probability p
            if random.uniform(0, 1) < p:
                random.shuffle(sentences)
                return ' '.join(sentences)
        
        return text_str

    @staticmethod
    def extract_sample_list(samples, shuffle_sentences: bool = False):
        """Extract sample list from raw data"""
        # Create extractor instance with shuffle parameter
        extractor = CTRATEExtractor(shuffle_sentences=shuffle_sentences)
        
        extracted_samples = []

        for sample in samples:
            # Classification samples for each condition
            classifications = sample["tasks"]["classification"]
            for condition in CTRATEExtractor.CONDITIONS:
                if condition in classifications:
                    value = classifications[condition]
                    if value is not None:  # Skip conditions with no value
                        extracted_samples.append(
                            {
                                "image_id": sample["image_id"],
                                # "image_path": sample["embedding_path"],
                                "question": f"Is there evidence of {condition} in this CT scan?",
                                "answer": "Yes" if value == 1 else "No",
                                "full_answer": "Yes" if value == 1 else "No",
                                # "full_answer": f"Based on my analysis of this CT scan, {'there is' if value == 1 else 'there is no'} evidence of {condition}.",
                                "answer_type": "CHOICE",
                                "answer_subtype": "BINARY",
                                "system_text": "You are a medical imaging AI trained to analyze CT scans.",
                            }
                        )

            # Combination of report and diagnosis
            report_text = extractor._shuffle_sentences(sample["tasks"]["report_gen"])
            diagnosis_text = extractor._shuffle_sentences(sample["tasks"]["diagnosis"])
            full_answer = f"Findings: {report_text}\nImpression: {diagnosis_text}"
            
            extracted_sample = {
                "image_id": sample["image_id"],
                # "image_path": sample["embedding_path"],
                "question": "Generate a detailed report and diagnosis for this CT scan.",
                "answer": full_answer,
                "full_answer": full_answer,
                "answer_type": "TEXT",
                "answer_subtype": "REPORT_DIAGNOSIS",
                "system_text": "You are a medical imaging AI trained to analyze CT scans.",
            }
            extracted_samples.append(extracted_sample)

            # Report generation sample
            if "report_gen" in sample["tasks"] and sample["tasks"]["report_gen"]:
                report_text = extractor._shuffle_sentences(sample["tasks"]["report_gen"])
                extracted_sample = {
                    "image_id": sample["image_id"],
                    # "image_path": sample["embedding_path"],
                    "question": "Generate a detailed report for this CT scan.",
                    "answer": report_text,
                    "full_answer": report_text,
                    "answer_type": "TEXT",
                    "answer_subtype": "REPORT",
                    "system_text": "You are a medical imaging AI trained to analyze CT scans.",
                }
                extracted_samples.append(extracted_sample)

            # Diagnosis sample
            if "diagnosis" in sample["tasks"] and sample["tasks"]["diagnosis"]:
                diagnosis_text = extractor._shuffle_sentences(sample["tasks"]["diagnosis"])
                extracted_sample = {
                    "image_id": sample["image_id"],
                    # "image_path": sample["embedding_path"],
                    "question": "What is the diagnosis for this CT scan?",
                    "answer": diagnosis_text,
                    "full_answer": diagnosis_text,
                    "answer_type": "TEXT",
                    "answer_subtype": "DIAGNOSIS",
                    "system_text": "You are a medical imaging AI trained to analyze CT scans.",
                }
                extracted_samples.append(extracted_sample)

            if "vqa_data" in sample:
                for qa_pair in sample["vqa_data"]:
                    answer_text = extractor._shuffle_sentences(qa_pair["answer"])
                    vqa_sample = {
                        "image_id": sample["image_id"],
                        # "image_path": sample["embedding_path"],
                        "question": qa_pair["question"],
                        "answer": answer_text,
                        "full_answer": answer_text,
                        "answer_type": "TEXT",
                        "answer_subtype": "VQA",
                        "system_text": "You are a medical imaging AI trained to analyze CT scans.",
                    }
                    extracted_samples.append(vqa_sample)  

        # convert to QAItemDict
        extracted_samples = [QAItemDict(sample) for sample in extracted_samples]

        return extracted_samples