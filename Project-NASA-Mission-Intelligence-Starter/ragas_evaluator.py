import os
import json
from typing import Dict, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy
from pathlib import Path

def _validate_inputs(
    question: str,
    answer: str,
    contexts: List[str],
) -> None:
    """Validate evaluation inputs."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Answer must be a non-empty string.")

    if not isinstance(contexts, list) or not contexts:
        raise ValueError("Contexts must be a non-empty list.")

    if not all(
        isinstance(context, str) and context.strip()
        for context in contexts
    ):
        raise ValueError(
            "Every context item must be a non-empty string."
        )


def _create_ragas_models():
    """Create RAGAS-compatible LLM and embedding models."""

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required."
        )

    evaluator_llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )

    evaluator_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        base_url=base_url,
    )

    ragas_llm = LangchainLLMWrapper(evaluator_llm)

    ragas_embeddings = LangchainEmbeddingsWrapper(
        evaluator_embeddings
    )

    return ragas_llm, ragas_embeddings


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
) -> Dict[str, float]:
    """
    Evaluate one RAG response using RAGAS.

    Returns Response Relevancy and Faithfulness scores.
    """

    try:
        _validate_inputs(
            question=question,
            answer=answer,
            contexts=contexts,
        )

        ragas_llm, ragas_embeddings = _create_ragas_models()

        sample = SingleTurnSample(
            user_input=question.strip(),
            response=answer.strip(),
            retrieved_contexts=[
                context.strip()
                for context in contexts
            ],
        )

        response_relevancy_metric = ResponseRelevancy(
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

        faithfulness_metric = Faithfulness(
            llm=ragas_llm,
        )

        response_relevancy = (
            response_relevancy_metric.single_turn_score(
                sample
            )
        )

        faithfulness = (
            faithfulness_metric.single_turn_score(
                sample
            )
        )

        return {
            "response_relevancy": float(
                response_relevancy
            ),
            "faithfulness": float(
                faithfulness
            ),
        }

    except Exception as exc:
        return {
            "error": str(exc)
        }
    
def load_evaluation_dataset(
    dataset_path: str = "evaluation_dataset.txt",
) -> List[Dict]:
    """Load and validate the evaluation dataset."""

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list) or not data:
        raise ValueError(
            "Evaluation dataset must contain a non-empty list."
        )

    required_fields = {
        "category",
        "mission",
        "question",
        "answer",
        "contexts",
    }

    for index, item in enumerate(data, start=1):

        if not isinstance(item, dict):
            raise ValueError(
                f"Dataset item {index} must be an object."
            )

        missing = required_fields - set(item.keys())

        if missing:
            raise ValueError(
                f"Dataset item {index} is missing fields: "
                f"{sorted(missing)}"
            )

    return data


def batch_evaluate(
    dataset_path: str = "evaluation_dataset.txt",
) -> Dict:
    """
    Evaluate all samples in the evaluation dataset.

    Returns per-question scores plus aggregate averages.
    """

    dataset = load_evaluation_dataset(dataset_path)

    results = []

    for index, item in enumerate(dataset, start=1):

        print(
            f"Evaluating {index}/{len(dataset)}: "
            f"{item['question']}"
        )

        scores = evaluate_response_quality(
            question=item["question"],
            answer=item["answer"],
            contexts=item["contexts"],
        )

        result = {
            "category": item["category"],
            "mission": item["mission"],
            "question": item["question"],
            **scores,
        }

        results.append(result)

    valid_results = [
        result
        for result in results
        if "error" not in result
    ]

    if not valid_results:
        return {
            "results": results,
            "aggregate": {},
        }

    aggregate = {
        "response_relevancy_mean": sum(
            result["response_relevancy"]
            for result in valid_results
        ) / len(valid_results),

        "faithfulness_mean": sum(
            result["faithfulness"]
            for result in valid_results
        ) / len(valid_results),

        "evaluated_questions": len(valid_results),
        "total_questions": len(results),
    }

    return {
        "results": results,
        "aggregate": aggregate,
    }