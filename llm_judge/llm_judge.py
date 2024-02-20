# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import re
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, ClassVar, Dict, List, Tuple, Union, Type, TypeVar, Callable
from enum import Enum
import mpi4py
import pandas as pd
import transformers

import mlrun
from mlrun.model import ModelObj
from mlrun.utils import logger
import openai
import json
import pathlib

# These prmopt are used to generate the grade for LLM-as a judge

"""
@misc{zheng2023judging,
      title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang
      and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li
      and Dacheng Li and Eric. P Xing and Hao Zhang and Joseph E. Gonzalez
      and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

SINGLE_GRADE_PROMPT = """
Task:
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user question displayed below. You will be given the definition of {name}, grading rubric, context information.
Your task is to determine a numerical score of {name} for the response. You must use the grading rubric to determine your score. You must also give a explanation about how did you determine the score step-by-step. Please use chain of thinking.
Examples could be included beblow for your reference. Make sure you understand the grading rubric and use the examples before completing the task.
[Examples]:
{examples}
[User Question]:
{question}
[Response]:
{answer}
[Definition of {name}]:
{definition}
[Grading Rubric]:
{rubric}
You must return the following fields in your output:
- score: a numerical score of {name} for the response
- explanation: a explanation about how did you determine the score step-by-step
[Output]:
"""

PAIR_GRADE_PROMPT = """
Task:
Your task is to determine two numerical score of {name} for the responses from two AI assistants. You must use the grading rubric to determine your scores. You must also give a explanation about how did you determine the scores step-by-step. Please using chain of thinking.
Examples could be included beblow for your reference. Make sure you understand the grading rubric and use the examples before completing the task.
[Examples]:
{examples}
[User Question]:
{question}
[Response of assistant A]:
{answerA}
[Response of assistant B]:
{answerB}
[Definition of {name}]:
{definition}
[Grading Rubric]:
{rubric}
You must return the following fields in your output:
- score of assistant a: a numerical score of {name} for the response
- explanation of assistant a: a explanation about how did you determine the score step-by-step
- score of assistant b: a numerical score of {name} for the response
- explanation of assistant b: a explanation about how did you determine the score step-by-step
[Output]:
"""

REF_GRADE_PROMPT = """
Task:
Your task is to determine two numerical score of {name} for the responses from two AI assistants with the ground truth of the response. You must use the grading rubric to determine your scores. You must use the ground truth of the response. You need to give a explanation about how did you compare with the ground truth of the response to determine the scores step-by-step. Please using chain of thinking.
Examples could be included beblow for your reference. Make sure you understand the grading rubric and use the examples before completing the task.
[Examples]:
{examples}
[User Question]:
{question}
[Response of assistant A]:
{answerA}
[Response of assistant B]:
{answerB}
[Ground truth of the response]:
{reference}
[Definition of {name}]:
{definition}
[Grading Rubric]:
{rubric}
You must return the following fields in your output:
- score of assistant a: a numerical score of {name} for the response
- explanation of assistant a: a explanation about how did you compare with the ground truth of the response to determine the score step-by-step
- score of assistant b: a numerical score of {name} for the response
- explanation of assistant b: a explanation about how did you compare with the ground truth of the response to determine the score step-by-step
[Output]:
"""


def _check_mlrun_and_open_mpi() -> Tuple["mlrun.MLClientCtx", "mpi4py.MPI.Intracomm"]:
    is_mpi = False
    try:
        context = mlrun.get_or_create_ctx(name="mlrun")
        is_mpi = context.labels.get("kind", "job") == "mpijob"

        if is_mpi:
            try:
                from mpi4py import MPI

                return context, MPI.COMM_WORLD
            except ModuleNotFoundError as mpi4py_not_found:
                logger.error(
                    "To distribute the function using MLRun's 'mpijob' you need to have `mpi4py` package in your "
                    "interpreter. Please run `pip install mpi4py` and make sure you have open-mpi."
                )
                raise mpi4py_not_found
    except ModuleNotFoundError as module_not_found:
        if is_mpi:
            raise module_not_found
    return None, None


def _open_mpi_handler(
    worker_inputs: str,
):
    # Check for MLRun and OpenMPI availability:
    context, comm = _check_mlrun_and_open_mpi()

    def _decorator(handler):
        if comm is None or comm.Get_size() == 1:
            return handler

        @wraps(handler)
        def _wrapper(**kwargs):
            # Get the open mpi environment properties:
            size = comm.Get_size()
            rank = comm.Get_rank()
            sample_df = kwargs[worker_inputs]

            # Give the correct chunk of the workers inputs:
            even_chunk_size = len(sample_df) // size
            chunk_start = rank * even_chunk_size
            chunk_end = (
                (rank + 1) * even_chunk_size if rank + 1 < size else len(sample_df)
            )
            logger.info(
                f"Rank #{rank}: Processing input chunk sample dataframe"
                f"from index {chunk_start} to {chunk_end}."
            )
            sample_df = sample_df.iloc[chunk_start:chunk_end:, :]
            kwargs[worker_inputs] = sample_df

            # Run the worker:
            output = handler(**kwargs)

            # Send the output to the root rank (rank #0):
            output = comm.gather(output, root=0)
            if rank == 0:
                # Join the outputs:
                logger.info("Collecting data from workers to root worker.")
                dataframe = pd.concat(objs=[df for df, _ in output], axis=0)
                return dataframe
            return None

        return _wrapper

    return _decorator


class LLMJudgeBaseMetric(ModelObj, ABC):
    """
    Base class of the metrics that computed by LLM as a judge
    We don't need the y_true as reference. These metrics are used for more open-ended question for the model
    and the algorithm is based on the paper https://arxiv.org/pdf/2306.05685.pdf
    """

    _dict_fields = [
        "name",
        "model_judge",
        "prompt_template",
        "prompt_config",
        "model_judge_config",
        "tokenizer_judge_config",
        "model_judge_infer_config",
    ]
    kind = "llm_judge_metric"
    default_name: ClassVar[str] = "llm_judge_metric"

    def __init__(
        self,
        name: str,
        model_judge: str,
        prompt_template: str,
        prompt_config: Dict[str, Any],
        model_judge_config: Dict[str, Any] = None,
        tokenizer_judge_config: Dict[str, Any] = None,
        model_judge_infer_config: Dict[str, Any] = None,
    ):
        """
        These metrics are used to evaluate the model performance on a given dataset
        :param name: name of the metric
        :param model_judge: the model judge to use
        :param model_judge_config: the model judge config
        :param tokenizer_judge_config: the tokenizer judge config
        :param model_judge_infer_config: the model judge infer config
        :param prompt_template: the prompt template to fill
        :param prompt_config: the prompt config to fill the template with
        """
        self.name = name or self.default_name
        self.model_judge = model_judge
        self.model_judge_config = model_judge_config
        self.tokenizer_judge_config = tokenizer_judge_config
        self.model_judge_infer_config = model_judge_infer_config
        self.prompt_template = prompt_template
        self.prompt_config = prompt_config

    def _fill_prompt(self) -> str:
        """
        Fill the prompt template with the prompt config
        :param prompt_template: the prompt template to fill
        :param prompt_config: the prompt config to fill the template with
        :returns: the filled prompt
        """
        logger.info("Filling the prompt template with the prompt config")
        return self.prompt_template.format(**self.prompt_config)

    @abstractmethod
    def _prepare_judge(self) -> None:
        """
        Prepare the judge model
        """
        pass

    @abstractmethod
    def _compute_over_one_data(self, question: str, response: str) -> Dict[str, Any]:
        """
        Compute the metrics over one data point
        :param question: the question to compute the metrics over
        :param response: the response to compute the metrics over
        :returns: the metrics score and the explanation
        """
        pass

    @abstractmethod
    def _compute_over_data(self, sample_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the metrics over one data point
        :param sample_df: the sample dataframe to compute the metrics over
        :returns: the metrics score and the explanation
        """
        pass

    @abstractmethod
    def _extract_score_explanation(self, result: str) -> Dict[str, Any]:
        """
        Abstract the store of the result
        :param result: the result text
        :returns: the stored result
        """
        pass


class LLMJudgeSingleGrading(LLMJudgeBaseMetric):
    """
    Base class for LLM as a judge using single grading.
    you need to define the defnition of the metrics and give the grading of the rubic
    """

    _dict_fields = [
        "name",
        "model_judge",
        "model_judge_config",
        "model_judge_infer_config",
        "prompt_config",
        "tokenizer_judge_config",
        "prompt_template",
    ]
    kind = "llm_judge_single_grading"

    def __init__(
        self,
        name: str,
        model_judge: str,
        model_judge_config: Dict[str, Any],
        model_judge_infer_config: Dict[str, Any],
        prompt_config: Dict[str, Any],
        prompt_template: str = SINGLE_GRADE_PROMPT,
        tokenizer_judge_config: Dict[str, Any] = None,
    ):
        """
        init the class
        :param name: name of the metric
        :param model_judge: the model judge to use
        :param model_judge_config: the model judge config
        :param tokenizer_judge_config: the tokenizer judge config
        :param model_judge_infer_config: the model judge infer config
        :param prompt_template: the prompt template to fill
        :param prompt_config: the prompt config to fill the template with
        """
        super().__init__(
            name,
            model_judge,
            prompt_template,
            prompt_config,
            model_judge_config,
            tokenizer_judge_config,
            model_judge_infer_config,
        )

    def _prepare_judge(self) -> None:
        """
        Prepare the judge model it will init the tokenizer and the model
        """
        logger.info(f"Preparing the judge model {self.model_judge}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_judge, **self.tokenizer_judge_config
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_judge, **self.model_judge_config
        )

    def _compute_over_one_data(self, question, response) -> Dict[str, Any]:
        """
        Compute the metrics over one data point
        :param question: the question to compute the metrics over
        :param response: the response to compute the metrics over
        :returns: the metrics score and the explanation
        """
        logger.info(
            f"Computing the metrics over one data point with {question} and {response}"
        )
        self.prompt_config["question"] = question
        self.prompt_config["answer"] = response
        input_ids = self.tokenizer(self._fill_prompt(), return_tensors="pt").input_ids
        outputs = self.model.generate(
            input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.model_judge_infer_config,
        )

        response_ids = outputs[0]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        res_dic = self._extract_score_explanation(response)
        return res_dic

    @_open_mpi_handler(worker_inputs="sample_df")
    def _compute_over_data(
        self,
        sample_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute the metrics over all data
        :param sample_df: the sample dataframe
        :returns: the metrics score and the explanation
        """
        self._prepare_judge()
        res_df = pd.DataFrame(columns=["question", "answer", "score", "explanation"])

        logger.info("Computing the metrics over all data")
        for i in range(len(sample_df)):
            res_dic = self._compute_over_one_data(
                sample_df.loc[i, "question"], sample_df.loc[i, "answer"]
            )
            res_df.loc[i] = [
                sample_df.loc[i, "question"],
                sample_df.loc[i, "answer"],
                res_dic["score"],
                res_dic["explanation"],
            ]

        return res_df

    def _extract_score_explanation(self, result: str) -> Dict[str, Any]:
        """
        Abstract the store of the result
        :param result: the result to store
        :returns: the stored result
        """
        logger.info(f"Extracting the score and explanation from {result}")
        score_pattern = r"\bscore:\s*(\d+)\b"
        explanation_pattern = r"explanation:\s*(.*?)\s*(?=\bScore:|$)"

        score_match = re.search(score_pattern, result)
        score = int(score_match.group(1)) if score_match else None

        explanation_match = re.search(explanation_pattern, result, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else None

        return {"score": score, "explanation": explanation}


class LLMJudgePairwiseGrading(LLMJudgeBaseMetric):
    """
    Base class for LLM as a judge using pairwise grading.
    you need to define the defnition of the metrics and give the grading of the rubic
    you need to give a base model to compare the model to
    """

    _dict_fields = [
        "name",
        "model_judge",
        "model_judge_config",
        "model_bench_mark",
        "model_bench_mark_config",
        "model_bench_mark_infer_config",
        "tokenizer_bench_mark_config",
        "prompt_template",
        "prompt_config",
        "tokenizer_judge_config",
        "model_judge_infer_config",
    ]
    kind = "llm_judge_pairwise_grading"

    def __init__(
        self,
        name: str,
        model_judge: str,
        model_judge_config: Dict[str, Any],
        model_bench_mark: str,
        model_bench_mark_config: Dict[str, Any],
        model_bench_mark_infer_config: Dict[str, Any],
        tokenizer_bench_mark_config: Dict[str, Any],
        prompt_config: Dict[str, Any],
        prompt_template: str = PAIR_GRADE_PROMPT,
        tokenizer_judge_config: Dict[str, Any] = None,
        model_judge_infer_config: Dict[str, Any] = None,
    ):
        """
        init the class
        :param name: name of the metric
        :param model_judge: the model judge to use
        :param tokenizer_judge_config: the tokenizer judge config
        :param model_judge_config: the model judge config
        :param model_judge_infer_config: the model judge infer config
        :param model_bench_mark: the model bench mark to use
        :param model_bench_mark_config: the model bench mark config
        :param model_bench_mark_infer_config: the model bench mark infer config
        :param tokenizer_bench_mark_config: the tokenizer bench mark config
        :param prompt_template: the prompt template to fill
        :param prompt_config: the prompt config to fill the template with
        """
        super().__init__(
            name,
            model_judge,
            prompt_template,
            prompt_config,
            model_judge_config,
            tokenizer_judge_config,
            model_judge_infer_config,
        )
        self.model_bench_mark = model_bench_mark
        self.model_bench_mark_config = model_bench_mark_config
        self.model_bench_mark_infer_config = model_bench_mark_infer_config
        self.tokenizer_bench_mark_config = tokenizer_bench_mark_config

    def _prepare_judge(self) -> None:
        """
        init the tokenizer and the model
        """
        logger.info(f"Preparing the judge model {self.model_judge}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_judge, **self.tokenizer_judge_config
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_judge, **self.model_judge_config
        )

    def _prepare_bench_mark_model(self) -> None:
        """
        Prepare the model that used for bench marking
        """
        logger.info(f"Preparing the bench mark model {self.model_bench_mark}")
        self.tokenizer_bench_mark = transformers.AutoTokenizer.from_pretrained(
            self.model_bench_mark, **self.tokenizer_bench_mark_config
        )
        self.model_bench_mark = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_bench_mark, **self.model_bench_mark_config
        )

    def _compute_bench_mark_response(self, question) -> str:
        """
        Compute the response of the bench mark model
        :param question: the question to ask the model
        :returns: the response
        """
        logger.info(f"Computing the bench mark response for {question}")
        input_ids = self.tokenizer_bench_mark(question, return_tensors="pt").input_ids
        outputs = self.model_bench_mark.generate(
            input_ids,
            pad_token_id=self.tokenizer_bench_mark.pad_token_id,
            eos_token_id=self.tokenizer_bench_mark.eos_token_id,
            **self.model_bench_mark_infer_config,
        )

        response_ids = outputs[0]
        response = self.tokenizer_bench_mark.decode(
            response_ids, skip_special_tokens=True
        )
        logger.info(f"Response of the bench mark model is {response}")

        return response

    def _compute_over_one_data(self, question, response) -> Dict[str, Any]:
        """
        Compute the metrics over one data point
        :param question
        :returns: the metrics score and the explanation
        """
        logger.info(f"Computing the metrics over {question} and {response}")
        self.prompt_config["question"] = question
        self.prompt_config["answerA"] = response
        self.prompt_config["answerB"] = self._compute_bench_mark_response(question)
        input_ids = self.tokenizer(self._fill_prompt(), return_tensors="pt").input_ids
        outputs = self.model.generate(
            input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.model_judge_infer_config,
        )

        response_ids = outputs[0]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        logger.info(f"Response of the judge model is {response}")
        res_dic = self._extract_score_explanation(response)
        res_dic["answerB"] = self.prompt_config["answerB"]
        return res_dic

    @_open_mpi_handler(worker_inputs="sample_df")
    def _compute_over_data(
        self,
        sample_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute the metrics over all data
        :param sample_df: the sample dataframe
        :returns: the metrics score and the explanation
        """
        self._prepare_judge()
        self._prepare_bench_mark_model()
        res_df = pd.DataFrame(
            columns=[
                "question",
                "answerA",
                "answerB",
                "score_of_assistant_a",
                "explanation_of_assistant_a",
                "score_of_assistant_b",
                "explanation_of_assistant_b",
            ]
        )

        for i in range(len(sample_df)):
            res_dic = self._compute_over_one_data(
                sample_df.loc[i, "question"],
                sample_df.loc[i, "answer"],
            )
            res_df.loc[i] = [
                sample_df.loc[i, "question"],
                sample_df.loc[i, "answer"],
                res_dic["answerB"],
                res_dic["score_of_assistant_a"],
                res_dic["explanation_of_assistant_a"],
                res_dic["score_of_assistant_b"],
                res_dic["explanation_of_assistant_b"],
            ]

        return res_df

    def _extract_score_explanation(self, response) -> Dict[str, Any]:
        """
        Extract the score and the explanation from the response
        :param response: the response to extract the score and the explanation from
        :returns: the score and the explanation
        """
        # Find the position of the "[Output]:" marker
        output_marker_index = response.find("[Output]:")
        if output_marker_index == -1:
            return "No '[Output]:' marker found"

        # Extract the part of the response after the "[Output]:" marker
        response_after_output = response[output_marker_index + len("[Output]:") :]

        # Adjusted pattern to match the text format and separate lines
        pattern = r"- score of assistant ([abAB]): (\d)\s*- explanation of assistant \1: (.*?)\s*(?=- score of assistant|$)"
        matches = re.findall(pattern, response_after_output, re.DOTALL)

        if matches:
            result_dict = {}
            for match in matches:
                assistant, score, explanation = match
                result_dict[f"score_of_assistant_{assistant}".lower()] = int(score)
                result_dict[
                    f"explanation_of_assistant_{assistant}".lower()
                ] = explanation.strip()
            return result_dict
        else:
            raise ValueError(
                "No matches found after '[Output]:' marker. "
                "Please check the format of the response."
            )


class LLMJudgeReferenceGrading(LLMJudgePairwiseGrading):
    """
    LLM Judge Reference Grading class
    you need to give the name of the metrics, give the grading rubric and the bench mark model to use
    This class requrie you know the y_true of the response
    """

    _dict_fields = [
        "name",
        "model_judge",
        "model_judge_config",
        "model_judge_infer_config",
        "tokenizer_judge_config",
        "model_bench_mark",
        "model_bench_mark_config",
        "model_bench_mark_infer_config",
        "tokenizer_bench_mark_config",
        "prompt_template",
        "prompt_config",
    ]
    kind = "llm_judge_reference_grading"

    def __init__(
        self,
        name: str,
        model_judge: str,
        model_judge_config: Dict[str, Any],
        model_judge_infer_config: Dict[str, Any],
        tokenizer_judge_config: Dict[str, Any],
        model_bench_mark: str,
        model_bench_mark_config: Dict[str, Any],
        tokenizer_bench_mark_config: Dict[str, Any],
        model_bench_mark_infer_config: Dict[str, Any],
        prompt_config: Dict[str, str],
        prompt_template: str = REF_GRADE_PROMPT,
    ):
        """
        init the grading with reference class
        :param name: the name of the metrics
        :param model_judge: the model to use for grading
        :param model_judge_config: the config of the model to use for grading
        :param model_judge_infer_config: the config of the model to use for inference
        :param tokenizer_judge_config: the config of the tokenizer to use for grading
        :param model_bench_mark: the model to use for bench marking
        :param model_bench_mark_config: the config of the model to use for bench marking
        :param tokenizer_bench_mark_config: the config of the tokenizer to use for bench marking
        :param model_bench_mark_infer_config: the config of the model to use for inference
        :param prompt_template: the template of the prompt to use
        :param prompt_config: the config of the prompt to use
        """
        super().__init__(
            name,
            model_judge,
            model_judge_config,
            model_bench_mark,
            model_bench_mark_config,
            model_bench_mark_infer_config,
            tokenizer_bench_mark_config,
            prompt_config,
            prompt_template,
            tokenizer_judge_config,
            model_judge_infer_config,
        )

    def _compute_over_one_data(self, question, response, reference) -> Dict[str, Any]:
        """
        Compute the metrics over one data point
        :param kwargs: the data to compute the metrics over
        :returns: the metrics score and the explanation
        """
        self.prompt_config["reference"] = reference
        res_dic = super()._compute_over_one_data(question, response)
        return res_dic

    @_open_mpi_handler(worker_inputs="sample_df")
    def _compute_over_data(
        self,
        sample_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute the metrics over a dataset
        :param sample_df: the data to compute the metrics over
        :returns: the metrics score and the explanation
        """
        df = super()._compute_over_data(sample_df)
        df["reference"] = sample_df["reference"]
        return df


class OPENAIJudgeSingleGrading(LLMJudgeSingleGrading):
    """
    Using API to judge the response
    """

    _dict_fields = [
        "name",
        "model_judge",
        "model_judge_config",
        "prompt_template",
        "prompt_config",
        "model_judge_infer_config",
    ]
    kind = "OPENAI_judge_single_grading"

    def __init__(
        self,
        name: str,
        model_judge: str,
        prompt_config: Dict[str, str],
        prompt_template: str = SINGLE_GRADE_PROMPT,
        model_judge_infer_config: Dict[str, Any] = None,
        model_judge_config: Dict[str, Any] = None,
    ):
        """
        init the grading with reference class
        :param name: the name of the metrics
        :param model_judge: the model to use for grading
        :param model_judge_config: the config of the model to use for grading
        :param model_judge_infer_config: the config of the model to use for inference
        :param prompt_template: the template of the prompt to use
        :param prompt_config: the config of the prompt to use
        """
        super().__init__(
            name,
            model_judge,
            model_judge_config,
            model_judge_infer_config,
            prompt_config,
            prompt_template,
        )

    def _prepare_judge(self) -> None:
        """
        Prepare the judge model
        """
        logger.info("Prepare the openAI model as judge")

        if not self.model_judge_config:
            import os
            from dotenv import load_dotenv

            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_API_BASE")
            OPENAI_JUDGE_CONFIG = {
                "api_key": api_key,
                "base_url": base_url,
            }
            self.model_judge_config = OPENAI_JUDGE_CONFIG

        self.model = openai.OpenAI(
            api_key=self.model_judge_config["api_key"],
            base_url=self.model_judge_config["base_url"],
        )

    def _extract_score_explanation(self, result: str) -> Dict[str, Any]:
        """
        Abstract the store of the result
        :param result: the result to store
        :returns: the stored result
        """
        logger.info(f"Extracting the score and explanation from {result}")
        score_pattern = r'"score":\s*(\d+)'
        explanation_pattern = r'"explanation":\s*"([^"]+)"'

        score_match = re.search(score_pattern, result)
        score = int(score_match.group(1)) if score_match else None

        explanation_match = re.search(explanation_pattern, result, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else None

        return {"score": score, "explanation": explanation}

    def _compute_over_one_data(self, question, response) -> Dict[str, Any]:
        """
        Compute the metrics over one data point
        :param kwargs: the data to compute the metrics over
        :returns: the metrics score and the explanation
        """
        logger.info(f"Compute the metrics over one data point using openAI's model")
        self.prompt_config["answer"] = response
        self.prompt_config["question"] = question
        prompt = self._fill_prompt()
        res = self.model.chat.completions.create(
            model=self.model_judge, messages=[{"role": "user", "content": prompt}]
        )
        res_dic = self._extract_score_explanation(res.choices[0].message.content)
        return res_dic


class OPENAIJudgePairwiseGrading(LLMJudgePairwiseGrading):
    """
    Using API to judge the response
    """

    _dict_fields = [
        "name",
        "model_judge",
        "model_judge_config",
        "model_bench_mark",
        "model_bench_mark_config",
        "model_bench_mark_infer_config",
        "tokenizer_bench_mark_config",
        "prompt_template",
        "prompt_config",
        "model_judge_infer_config",
    ]
    kind = "OPENAI_judge_pair_grading"

    def __init__(
        self,
        name: str,
        model_judge: str,
        model_bench_mark: str,
        model_bench_mark_config: Dict[str, Any],
        model_bench_mark_infer_config: Dict[str, Any],
        tokenizer_bench_mark_config: Dict[str, Any],
        prompt_config: Dict[str, str],
        model_judge_config: Dict[str, Any] = None,
        model_judge_infer_config: Dict[str, Any] = None,
        prompt_template: str = PAIR_GRADE_PROMPT,
    ):
        """
        init the grading with reference class
        :param name: the name of the metrics
        :param model_judge: the model to use for grading
        :param model_judge_config: the config of the model to use for grading
        :param model_judge_infer_config: the config of the model to use for inference
        :param prompt_template: the template of the prompt to use
        :param prompt_config: the config of the prompt to use
        """
        super().__init__(
            name,
            model_judge,
            model_judge_config,
            model_bench_mark,
            model_bench_mark_config,
            model_bench_mark_infer_config,
            tokenizer_bench_mark_config,
            prompt_config,
            prompt_template,
            model_judge_infer_config,
        )

    def _prepare_judge(self) -> None:
        """
        Prepare the judge model
        """
        logger.info("Prepare the openAI model as judge")
        if not self.model_judge_config:
            import os
            from dotenv import load_dotenv

            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_API_BASE")
            OPENAI_JUDGE_CONFIG = {
                "api_key": api_key,
                "base_url": base_url,
            }
            self.model_judge_config = OPENAI_JUDGE_CONFIG
        self.model = openai.OpenAI(
            api_key=self.model_judge_config["api_key"],
            base_url=self.model_judge_config["base_url"],
        )

    def _compute_over_one_data(self, question, response) -> Dict[str, Any]:
        """
        Compute the metrics over one data point
        :param kwargs: the data to compute the metrics over
        :returns: the metrics score and the explanation
        """
        logger.info(f"Computing the metrics over {question} and {response}")
        self.prompt_config["question"] = question
        self.prompt_config["answerA"] = response
        self.prompt_config["answerB"] = self._compute_bench_mark_response(question)
        prompt = self._fill_prompt()
        res = self.model.chat.completions.create(
            model=self.model_judge,
            messages=[{"role": "user", "content": prompt}],
        )
        res_dic = self._extract_score_explanation(res.choices[0].message.content)
        res_dic["answerB"] = self.prompt_config["answerB"]
        return res_dic

    def _extract_score_explanation(self, response) -> Dict[str, Any]:
        """
        Extract the score and the explanation from the response
        :param response: the response to extract the score and the explanation from
        :returns: the score and the explanation
        """
        try:
            res = json.loads(response)
            result_dict = {}
            result_dict["score_of_assistant_a"] = res["score of assistant a"]
            result_dict["score_of_assistant_b"] = res["score of assistant b"]
            result_dict["explanation_of_assistant_a"] = res[
                "explanation of assistant a"
            ]
            result_dict["explanation_of_assistant_b"] = res[
                "explanation of assistant b"
            ]
            return result_dict
        except Exception as e:
            # Adjusted pattern to match the text format and separate lines
            pattern = r"-?\s?score of assistant ([a-zA-Z]+): (\d+).*?-?\s?explanation of assistant [a-zA-Z]+: (.*?)(?=-?\s?score of assistant [a-zA-Z]+:|$)"
            matches = re.findall(pattern, response, re.DOTALL)

            if matches:
                result_dict = {}
                for match in matches:
                    assistant, score, explanation = match
                    result_dict[f"score_of_assistant_{assistant}".lower()] = int(score)
                    result_dict[
                        f"explanation_of_assistant_{assistant}".lower()
                    ] = explanation.strip()
                return result_dict
            else:
                raise ValueError(
                    "No matches found after '[Output]:' marker. "
                    "Please check the format of the response."
                )


class OPENAIJudgeReferenceGrading(OPENAIJudgePairwiseGrading, LLMJudgeReferenceGrading):
    """
    OPENAI Judge Reference Grading class
    you need to give the name of the metrics, give the grading rubric and the bench mark model to use
    This class requrie you know the y_true of the response
    """

    _dict_fields = [
        "name",
        "model_judge",
        "model_judge_config",
        "model_bench_mark",
        "model_bench_mark_config",
        "model_bench_mark_infer_config",
        "tokenizer_bench_mark_config",
        "prompt_template",
        "prompt_config",
        "model_judge_infer_config",
    ]
    kind = "OPENAI_judge_reference_grading"

    def __init__(
        self,
        name: str,
        model_judge: str,
        model_bench_mark: str,
        model_bench_mark_config: Dict[str, Any],
        tokenizer_bench_mark_config: Dict[str, Any],
        model_bench_mark_infer_config: Dict[str, Any],
        prompt_config: Dict[str, str],
        model_judge_config: Dict[str, Any] = None,
        prompt_template: str = REF_GRADE_PROMPT,
        model_judge_infer_config: Dict[str, Any] = None,
    ):
        """
        init the grading with reference class
        :param name: the name of the metrics
        :param model_judge: the model to use for grading
        :param model_judge_config: the config of the model to use for grading
        :param model_judge_infer_config: the config of the model to use for inference
        :param model_bench_mark: the model to use for bench marking
        :param model_bench_mark_config: the config of the model to use for bench marking
        :param tokenizer_bench_mark_config: the config of the tokenizer to use for bench marking
        :param model_bench_mark_infer_config: the config of the model to use for inference
        :param prompt_template: the template of the prompt to use
        :param prompt_config: the config of the prompt to use
        """
        super().__init__(
            name,
            model_judge,
            model_bench_mark,
            model_bench_mark_config,
            model_bench_mark_infer_config,
            tokenizer_bench_mark_config,
            prompt_config,
            model_judge_config,
            model_judge_infer_config,
            prompt_template,
        )

    def _compute_over_one_data(self, question, response, reference) -> Dict[str, Any]:
        """
        Compute the metrics over one data point
        :param kwargs: the data to compute the metrics over
        :returns: the metrics score and the explanation
        """
        self.prompt_config["reference"] = reference
        res_dic = super()._compute_over_one_data(question, response)
        return res_dic

    @_open_mpi_handler(worker_inputs="sample_df")
    def _compute_over_data(self, sample_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the metrics over a dataset
        :param sample_df: the data to compute the metrics over
        :returns: the metrics score and the explanation
        """
        return LLMJudgeReferenceGrading._compute_over_data(self, sample_df)


MetricsType_dic = {
    "LLMJudgeSingleGrading": LLMJudgeSingleGrading,
    "LLMJudgePairwiseGrading": LLMJudgePairwiseGrading,
    "LLMJudgeReferenceGrading": LLMJudgeReferenceGrading,
    "OPENAIJudgeSingleGrading": OPENAIJudgeSingleGrading,
    "OPENAIJudgePairwiseGrading": OPENAIJudgePairwiseGrading,
    "OPENAIJudgeReferenceGrading": OPENAIJudgeReferenceGrading,
}

MetricsType = TypeVar(
    "MetricsType",
    LLMJudgeSingleGrading,
    LLMJudgePairwiseGrading,
    LLMJudgeReferenceGrading,
    OPENAIJudgeSingleGrading,
    OPENAIJudgePairwiseGrading,
    OPENAIJudgeReferenceGrading,
)


def _get_metrics(
    metric_type: str,
    **kwargs: Any,
) -> MetricsType:
    """
    Init the metric class based on different type of metrics
    :param metric_type: the type of the metric
    :param kwargs: the config of the metric
    :returns: the metric obj
    """
    return MetricsType_dic[metric_type](**kwargs)


def llm_judge(
    input_path: Union[str, pathlib.Path],
    **kwargs,
) -> pd.DataFrame:
    """
    Compute the metrics over a dataset

    :param input_path: the path to the input data
    :param kwargs: the config of the metric

    :returns: the metrics score and the explanation
    """
    metric = _get_metrics(**kwargs)
    sample_df = pd.read_csv(input_path)
    res_df = metric._compute_over_data(sample_df)
    return res_df
