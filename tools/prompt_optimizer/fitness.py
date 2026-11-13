"""
Fitness calculation with anti-overfitting penalties.
"""

import statistics
from dataclasses import dataclass

from tools.prompt_optimizer.config import FitnessConfig
from tools.prompt_optimizer.llm_adapter import EvaluationResult


@dataclass
class FitnessScore:
    """Detailed fitness score breakdown."""
    base_score: float
    variance_penalty: float
    length_penalty: float
    specificity_penalty: float
    generalization_gap_penalty: float
    final_fitness: float

    # Component scores
    accuracy: float
    fluency: float
    style: float
    overall: float

    # Statistics
    train_mean: float
    test_mean: float
    score_variance: float
    prompt_length: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'base_score': self.base_score,
            'variance_penalty': self.variance_penalty,
            'length_penalty': self.length_penalty,
            'specificity_penalty': self.specificity_penalty,
            'generalization_gap_penalty': self.generalization_gap_penalty,
            'final_fitness': self.final_fitness,
            'accuracy': self.accuracy,
            'fluency': self.fluency,
            'style': self.style,
            'overall': self.overall,
            'train_mean': self.train_mean,
            'test_mean': self.test_mean,
            'score_variance': self.score_variance,
            'prompt_length': self.prompt_length
        }


# Text-specific terms that might indicate overfitting
TEXT_SPECIFIC_TERMS = [
    # Author names
    'austen', 'wilde', 'doyle', 'thoreau', 'melville',
    # Book titles
    'pride', 'prejudice', 'dorian', 'gray', 'scarlet',
    'walden', 'moby', 'dick',
    # Character names
    'ishmael', 'watson', 'holmes', 'wotton',
    # Specific references
    'afghanistan', 'laburnum', 'persian',
]


class FitnessCalculator:
    """
    Calculates fitness scores with anti-overfitting penalties.

    Formula:
    FITNESS = BASE_SCORE - PENALTIES

    BASE_SCORE = accuracy*w1 + fluency*w2 + style*w3 + overall*w4

    PENALTIES:
    - variance_penalty: variance(scores) * penalty_weight
    - length_penalty: max(0, length-threshold) * rate
    - specificity_penalty: count(text_specific_terms) * rate
    - generalization_gap: max(0, train_score - test_score) * weight
    """

    def __init__(self, config: FitnessConfig):
        """
        Initialize the fitness calculator.

        Args:
            config: Fitness configuration with weights and penalties
        """
        self.config = config

    def calculate_base_score(self, evaluations: list[EvaluationResult]) -> tuple[float, float, float, float, float]:
        """
        Calculate base score from evaluation results.

        Args:
            evaluations: List of evaluation results

        Returns:
            Tuple of (base_score, accuracy, fluency, style, overall)
        """
        if not evaluations:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        # Filter successful evaluations
        valid_evals = [e for e in evaluations if e.success]
        if not valid_evals:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        # Calculate means
        accuracy = statistics.mean(e.accuracy for e in valid_evals)
        fluency = statistics.mean(e.fluency for e in valid_evals)
        style = statistics.mean(e.style for e in valid_evals)
        overall = statistics.mean(e.overall for e in valid_evals)

        # Weighted score
        base_score = (
            accuracy * self.config.accuracy_weight +
            fluency * self.config.fluency_weight +
            style * self.config.style_weight +
            overall * self.config.overall_weight
        )

        return base_score, accuracy, fluency, style, overall

    def calculate_variance_penalty(self, evaluations: list[EvaluationResult]) -> tuple[float, float]:
        """
        Calculate variance penalty (penalizes inconsistent performance).

        Args:
            evaluations: List of evaluation results

        Returns:
            Tuple of (penalty, variance)
        """
        valid_evals = [e for e in evaluations if e.success]
        if len(valid_evals) < 2:
            return 0.0, 0.0

        # Calculate variance of weighted scores
        scores = [e.weighted_score for e in valid_evals]
        variance = statistics.variance(scores)
        penalty = variance * self.config.variance_penalty_weight

        return penalty, variance

    def calculate_length_penalty(self, prompt_text: str) -> tuple[float, int]:
        """
        Calculate length penalty (penalizes overly long prompts).

        Args:
            prompt_text: The full prompt text

        Returns:
            Tuple of (penalty, length)
        """
        length = len(prompt_text)
        excess = max(0, length - self.config.length_penalty_threshold)
        penalty = excess * self.config.length_penalty_rate

        return penalty, length

    def calculate_specificity_penalty(self, prompt_text: str) -> tuple[float, int]:
        """
        Calculate specificity penalty (penalizes text-specific terms).

        Args:
            prompt_text: The full prompt text

        Returns:
            Tuple of (penalty, count)
        """
        prompt_lower = prompt_text.lower()
        count = sum(1 for term in TEXT_SPECIFIC_TERMS if term in prompt_lower)
        penalty = count * self.config.specificity_penalty_rate

        return penalty, count

    def calculate_generalization_gap(
        self,
        train_score: float,
        test_score: float
    ) -> float:
        """
        Calculate generalization gap penalty.

        A large gap between train and test scores indicates overfitting.

        Args:
            train_score: Mean score on training set
            test_score: Mean score on test set

        Returns:
            Penalty value
        """
        gap = max(0.0, train_score - test_score)
        return gap * self.config.generalization_gap_weight

    def calculate_fitness(
        self,
        train_evaluations: list[EvaluationResult],
        test_evaluations: list[EvaluationResult],
        prompt_text: str
    ) -> FitnessScore:
        """
        Calculate complete fitness score with all penalties.

        Args:
            train_evaluations: Evaluations on training set
            test_evaluations: Evaluations on test set
            prompt_text: The full prompt text (system + user)

        Returns:
            FitnessScore with detailed breakdown
        """
        # Base score from training evaluations
        base_score, accuracy, fluency, style, overall = self.calculate_base_score(train_evaluations)

        # Calculate test score for generalization gap
        test_base, _, _, _, _ = self.calculate_base_score(test_evaluations)

        # Calculate penalties
        variance_penalty, variance = self.calculate_variance_penalty(train_evaluations + test_evaluations)
        length_penalty, prompt_length = self.calculate_length_penalty(prompt_text)
        specificity_penalty, _ = self.calculate_specificity_penalty(prompt_text)
        gap_penalty = self.calculate_generalization_gap(base_score, test_base)

        # Final fitness
        total_penalty = variance_penalty + length_penalty + specificity_penalty + gap_penalty
        final_fitness = base_score - total_penalty

        return FitnessScore(
            base_score=base_score,
            variance_penalty=variance_penalty,
            length_penalty=length_penalty,
            specificity_penalty=specificity_penalty,
            generalization_gap_penalty=gap_penalty,
            final_fitness=final_fitness,
            accuracy=accuracy,
            fluency=fluency,
            style=style,
            overall=overall,
            train_mean=base_score,
            test_mean=test_base,
            score_variance=variance,
            prompt_length=prompt_length
        )

    def calculate_quick_fitness(
        self,
        evaluations: list[EvaluationResult],
        prompt_text: str
    ) -> float:
        """
        Calculate a quick fitness score (no train/test split).

        Useful for initial population evaluation.

        Args:
            evaluations: All evaluation results
            prompt_text: The full prompt text

        Returns:
            Quick fitness score
        """
        base_score, _, _, _, _ = self.calculate_base_score(evaluations)

        # Only apply length and specificity penalties
        length_penalty, _ = self.calculate_length_penalty(prompt_text)
        specificity_penalty, _ = self.calculate_specificity_penalty(prompt_text)

        return base_score - length_penalty - specificity_penalty


def rank_prompts_by_fitness(
    prompts_with_scores: list[tuple[any, FitnessScore]]
) -> list[tuple[any, FitnessScore, int]]:
    """
    Rank prompts by fitness score.

    Args:
        prompts_with_scores: List of (prompt, fitness_score) tuples

    Returns:
        List of (prompt, fitness_score, rank) tuples, sorted by fitness (best first)
    """
    sorted_prompts = sorted(
        prompts_with_scores,
        key=lambda x: x[1].final_fitness,
        reverse=True
    )

    return [
        (prompt, score, rank + 1)
        for rank, (prompt, score) in enumerate(sorted_prompts)
    ]


def fitness_summary(score: FitnessScore) -> str:
    """
    Generate a human-readable summary of a fitness score.

    Args:
        score: The fitness score to summarize

    Returns:
        Formatted string summary
    """
    lines = [
        f"Fitness: {score.final_fitness:.3f}",
        f"  Base Score: {score.base_score:.3f}",
        f"    Accuracy: {score.accuracy:.2f}",
        f"    Fluency: {score.fluency:.2f}",
        f"    Style: {score.style:.2f}",
        f"    Overall: {score.overall:.2f}",
        f"  Penalties:",
        f"    Variance: -{score.variance_penalty:.3f} (var={score.score_variance:.3f})",
        f"    Length: -{score.length_penalty:.3f} ({score.prompt_length} chars)",
        f"    Specificity: -{score.specificity_penalty:.3f}",
        f"    Gap: -{score.generalization_gap_penalty:.3f} (train={score.train_mean:.2f}, test={score.test_mean:.2f})",
    ]

    return "\n".join(lines)
