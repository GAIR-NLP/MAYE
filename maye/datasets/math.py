import re
from pathlib import Path

import math_verify
from PIL import Image
from torch.utils.data import Dataset

from maye.datasets.math_utils import AutoScoringJudge
from maye.utils.file import open_jsonl

CHAT_COT_PROMPT = "\nSolve the above mathematics problem, write out the solution process according to the question, and use the same LaTeX format as the question in the solution process. Please display the final answer in the format \\boxed{}."

ZERO_COT_PROMPT = "A conversation between User and Assistant. The user asks a question with an image,  and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the solution. The reasoning process is enclosed within '<think>' and '</think>', followed by the solution. Ensuring that the final answer in solution is enclosed in \\boxed{{}}. User: {{question}} <|vision_start|><|image_pad|><|vision_end|> Assistant:"


class MathGenerationDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        use_chat_template: bool,
    ):
        self.dataset_path = Path(dataset_path)
        self.ds = open_jsonl(dataset_path)
        self.scorer = AutoScoringJudge()
        self.use_chat_template = use_chat_template

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: int):
        sample = self.ds[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: dict) -> dict:
        images = [Image.open(self.dataset_path.parent / sample["image"])]

        if self.use_chat_template:
            prompt_str = sample["question"] + CHAT_COT_PROMPT
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": sample["image"]},
                        {"type": "text", "text": prompt_str},
                    ],
                }
            ]
        else:
            prompt = ZERO_COT_PROMPT.format(question=sample["question"])

        prepared_sample = dict(**sample, images=images, prompt=prompt)

        return prepared_sample

    def judge(self, prediction: str, solution: str):
        prediction, solution = self.preprocess(prediction), self.preprocess(solution)
        try:
            math_verify_judge = math_verify.verify(
                math_verify.parse(prediction), math_verify.parse(solution)
            )
        except:
            math_verify_judge = False
        try:
            scorer_judge = self.scorer.judge(prediction=prediction, answer=solution)
        except:
            scorer_judge = False

        pattern = r"\\boxed\{(\d*?)(?:\^|$)"  # number until "^"
        number_prediction_match = re.search(pattern, prediction)
        number_answer_match = re.search(pattern, solution)
        number_judge = False
        if number_prediction_match and number_answer_match:
            number_prediction = number_prediction_match.group(1)
            number_answer = number_answer_match.group(1)
            number_judge = number_prediction == number_answer

        return math_verify_judge or scorer_judge or number_judge

    def preprocess(self, text: str):
        def extract_last_boxed_sentence(text: str):
            sentences = re.split(r"(?<!\d)[.!?](?!\d)", text)
            boxed_sentences = []
            for sentence in sentences:
                cleaned_sentence = sentence.strip()
                if "boxed" in cleaned_sentence:
                    boxed_sentences.append(cleaned_sentence)
            if boxed_sentences:
                return boxed_sentences[-1]
            else:
                return ""

        def latex_main_body_rule_process(text: str):
            text = text.replace("\\text{circ}", "\\circ")
            text = text.replace("\\text{degree}", "\\circ")
            text = text.replace("\\{circ}", "\\circ")
            text = text.replace("\\{degree}", "\\circ")
            text = text.replace("\\textcirc", "\\circ")
            text = text.replace("\\textdegree", "\\circ")
            text = text.replace("\\text{circles}", "\\circ")
            text = text.replace("\\textcircles", "\\circ")
            text = text.replace("\\text{o}", "\\circ")
            text = text.replace("\\texto", "\\circ")
            text = text.replace("\\bigcirc", "\\circ")
            text = text.replace("\\bullet", "\\circ")
            text = text.replace("{sqrt}", "\\sqrt")
            text = text.replace("{radical}", "\\sqrt")
            text = text.replace("{°}", "\\circ")
            text = text.replace("\\,\\text{cm}", "cm")
            return text

        text = extract_last_boxed_sentence(text)
        text = latex_main_body_rule_process(text)
        return text
