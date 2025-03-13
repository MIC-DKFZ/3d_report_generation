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
        minimal_answer = "The answer is"

        return {
            "question": qa_pair["question"],
            "options": qa_pair["options"],
            "answer": qa_pair["answer"],
            "reasoning": reasoning,
            "correct_option_text": correct_option_text,
            "options_text": options_text,
            # Formatted texts
            "user_text": user_text,
            "full_answer": full_answer,
            "minimal_answer": minimal_answer,
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
            "user_text": f"What do you see in the {region} region of this CT scan?",
            "full_answer": findings,
            "minimal_answer": "",
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
