from typing import Dict, Any, List, Optional, Tuple


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
            "question": qa_pair["question"],
            "options": qa_pair["options"],
            "answer": qa_pair["answer"],
            "answer_type": "CHOICE",
            "reasoning": reasoning,
            "correct_option_text": correct_option_text,
            "options_text": options_text,
            # Formatted texts
            "user_text": user_text,
            "full_answer": full_answer,
            "system_text": "You are a medical imaging AI trained to answer questions about CT scans.",
        }

    @staticmethod
    def extract_region_question(region: str, findings: str) -> Dict[str, Any]:
        """
        Extract formatted texts for region questions.
        """
        if not findings:
            return None

        return {
            "region": region,
            "findings": findings,
            "answer_type": "OPEN",
            "user_text": f"What do you see in the {region} region of this CT scan?",
            "full_answer": findings,
            "system_text": "You are a medical imaging AI trained to analyze CT scans.",
        }

    @staticmethod
    def extract_report(report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all information from an AMOS-MM report.
        """
        extracted_report = {
            "raw": report,  # Keep the original data
            "regions": {},
            "region_questions": {},
            "impression": report.get("impression", ""),
        }

        # Extract region findings and create region questions
        if "findings" in report:
            for region in ["chest", "abdomen", "pelvis"]:
                findings = report["findings"].get(region, "")
                extracted_report["regions"][region] = findings

                # Create formatted region question
                if findings:
                    extracted_report["region_questions"][region] = AMOSMMExtractor.extract_region_question(
                        region, findings
                    )

        return extracted_report

    @staticmethod
    def extract_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all relevant information from an AMOS-MM sample.
        """
        extracted = {"image_path": sample["image"], "qa_items": [], "report": None}

        if "labels" in sample:
            if "report" in sample["labels"]:
                extracted["report"] = AMOSMMExtractor.extract_report(sample["labels"]["report"])

            if "qa" in sample["labels"]:
                for qa_pair in sample["labels"]["qa"]:
                    extracted["qa_items"].append(AMOSMMExtractor.extract_qa(qa_pair))

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
        # Basic information from the sample
        extracted = {
            "question": qa_item["question"],
            "answer": qa_item["answer"],
            "content_type": qa_item["content_type"],
            "answer_type": qa_item["answer_type"],
            "scan_id": qa_item["scan_id"],
            "series_uid": qa_item["series_uid"],
            "subject_id": qa_item["subject_id"],
            "study_uid": qa_item["study_uid"],
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
            "study_uid": sample["study_uid"],
            "item_qa": extracted_qa,
        }
