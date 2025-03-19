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
            "answer_subtype": "MULTIPLE_CHOICE",
            "reasoning": reasoning,
            "correct_option_text": correct_option_text,
            "options_text": options_text,
            # Formatted texts
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
        extracted = {"image_path": sample["image"], "qa_items": [], "report": {}, "questions": []}

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
            "content_type": qa_item["content_type"],
            "answer_type": qa_item["answer_type"],
            "answer_subtype": answer_subtype,
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
