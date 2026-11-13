"""
Persistence and history tracking for optimization results.
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import yaml

from tools.prompt_optimizer.prompt_template import PromptTemplate


@dataclass
class IterationResult:
    """Results from a single optimization iteration."""
    iteration: int
    fold: int
    population_size: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    best_prompt_id: str
    elapsed_seconds: float
    evaluation_cost: float = 0.0
    prompts_evaluated: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PromptResult:
    """Final result for a prompt."""
    rank: int
    id: str
    parent_id: Optional[str]
    generation: int
    fitness: float
    train_score: float
    test_score: float
    generalization_gap: float
    system_prompt: str
    user_prompt: str
    mutation_history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class OptimizationReport:
    """Complete optimization run report."""
    run_id: str
    start_time: str
    end_time: Optional[str]
    config_summary: dict
    total_iterations: int
    completed_iterations: int
    total_evaluations: int
    total_cost: float
    fitness_progression: list[float]
    iterations: list[IterationResult]
    top_prompts: list[PromptResult]
    status: str = "running"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'run_id': self.run_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'config_summary': self.config_summary,
            'total_iterations': self.total_iterations,
            'completed_iterations': self.completed_iterations,
            'total_evaluations': self.total_evaluations,
            'total_cost': self.total_cost,
            'fitness_progression': self.fitness_progression,
            'iterations': [i.to_dict() for i in self.iterations],
            'top_prompts': [p.to_dict() for p in self.top_prompts],
            'status': self.status
        }


class HistoryManager:
    """
    Manages persistence of optimization results.

    Directory structure:
    output_dir/
    ├── run_YYYYMMDD_HHMMSS/
    │   ├── iteration_001.json
    │   ├── iteration_002.json
    │   ├── ...
    │   ├── final_report.json
    │   └── best_prompts/
    │       ├── prompt_01.yaml
    │       ├── prompt_02.yaml
    │       └── ...
    """

    def __init__(self, output_dir: str):
        """
        Initialize the history manager.

        Args:
            output_dir: Base output directory
        """
        self.output_dir = Path(output_dir)
        self.run_id: Optional[str] = None
        self.run_dir: Optional[Path] = None
        self.report: Optional[OptimizationReport] = None

    def start_run(self, config_summary: dict, total_iterations: int) -> str:
        """
        Start a new optimization run.

        Args:
            config_summary: Summary of configuration
            total_iterations: Total planned iterations

        Returns:
            The run ID
        """
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.run_id}"

        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "best_prompts").mkdir(exist_ok=True)

        # Initialize report
        self.report = OptimizationReport(
            run_id=self.run_id,
            start_time=datetime.now().isoformat(),
            end_time=None,
            config_summary=config_summary,
            total_iterations=total_iterations,
            completed_iterations=0,
            total_evaluations=0,
            total_cost=0.0,
            fitness_progression=[],
            iterations=[],
            top_prompts=[],
            status="running"
        )

        self._save_report()
        return self.run_id

    def record_iteration(
        self,
        iteration: int,
        fold: int,
        population: list[PromptTemplate],
        elapsed_seconds: float,
        evaluation_cost: float = 0.0
    ) -> None:
        """
        Record results from an iteration.

        Args:
            iteration: Iteration number
            fold: Current fold number
            population: Population of prompts with fitness scores
            elapsed_seconds: Time taken for this iteration
            evaluation_cost: Cost of evaluations in this iteration
        """
        if not self.report or not self.run_dir:
            return

        # Calculate statistics
        fitness_scores = [p.fitness for p in population if p.fitness > 0]
        if not fitness_scores:
            return

        best_fitness = max(fitness_scores)
        mean_fitness = sum(fitness_scores) / len(fitness_scores)
        worst_fitness = min(fitness_scores)
        best_prompt = max(population, key=lambda p: p.fitness)

        # Create iteration result
        result = IterationResult(
            iteration=iteration,
            fold=fold,
            population_size=len(population),
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            worst_fitness=worst_fitness,
            best_prompt_id=best_prompt.id,
            elapsed_seconds=elapsed_seconds,
            evaluation_cost=evaluation_cost,
            prompts_evaluated=len(fitness_scores)
        )

        # Update report
        self.report.iterations.append(result)
        self.report.completed_iterations = iteration
        self.report.fitness_progression.append(best_fitness)
        self.report.total_cost += evaluation_cost

        # Save iteration file
        iteration_file = self.run_dir / f"iteration_{iteration:03d}.json"
        iteration_data = {
            'summary': result.to_dict(),
            'population': [p.to_dict() for p in population]
        }
        with open(iteration_file, 'w', encoding='utf-8') as f:
            json.dump(iteration_data, f, indent=2, ensure_ascii=False)

        # Update main report
        self._save_report()

    def finalize_run(
        self,
        population: list[PromptTemplate],
        total_evaluations: int,
        status: str = "completed"
    ) -> None:
        """
        Finalize the optimization run.

        Args:
            population: Final population
            total_evaluations: Total number of evaluations performed
            status: Final status (completed, failed, interrupted)
        """
        if not self.report or not self.run_dir:
            return

        # Sort population by fitness
        sorted_pop = sorted(population, key=lambda p: p.fitness, reverse=True)

        # Create top prompts list
        self.report.top_prompts = []
        for rank, prompt in enumerate(sorted_pop[:10], 1):  # Top 10
            train_score = sum(prompt.train_scores) / len(prompt.train_scores) if prompt.train_scores else 0
            test_score = sum(prompt.test_scores) / len(prompt.test_scores) if prompt.test_scores else 0

            result = PromptResult(
                rank=rank,
                id=prompt.id,
                parent_id=prompt.parent_id,
                generation=prompt.generation,
                fitness=prompt.fitness,
                train_score=train_score,
                test_score=test_score,
                generalization_gap=max(0, train_score - test_score),
                system_prompt=prompt.system_prompt,
                user_prompt=prompt.user_prompt,
                mutation_history=prompt.mutation_history
            )
            self.report.top_prompts.append(result)

        # Update report
        self.report.end_time = datetime.now().isoformat()
        self.report.total_evaluations = total_evaluations
        self.report.status = status

        # Save final report
        self._save_report()

        # Export best prompts as YAML
        self._export_best_prompts(sorted_pop[:5])  # Top 5

    def _save_report(self) -> None:
        """Save the current report to file."""
        if not self.report or not self.run_dir:
            return

        report_file = self.run_dir / "final_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.report.to_dict(), f, indent=2, ensure_ascii=False)

    def _export_best_prompts(self, prompts: list[PromptTemplate]) -> None:
        """Export best prompts as YAML files."""
        if not self.run_dir:
            return

        prompts_dir = self.run_dir / "best_prompts"

        for rank, prompt in enumerate(prompts, 1):
            train_score = sum(prompt.train_scores) / len(prompt.train_scores) if prompt.train_scores else 0
            test_score = sum(prompt.test_scores) / len(prompt.test_scores) if prompt.test_scores else 0

            prompt_data = {
                'id': prompt.id,
                'rank': rank,
                'fitness': round(prompt.fitness, 4),
                'train_score': round(train_score, 4),
                'test_score': round(test_score, 4),
                'generation': prompt.generation,
                'mutation_history': prompt.mutation_history,
                'system_prompt': prompt.system_prompt,
                'user_prompt': prompt.user_prompt
            }

            prompt_file = prompts_dir / f"prompt_{rank:02d}.yaml"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                yaml.dump(prompt_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def get_progress_summary(self) -> str:
        """
        Get a summary of current progress.

        Returns:
            Formatted progress string
        """
        if not self.report:
            return "No run in progress"

        lines = [
            f"Run: {self.run_id}",
            f"Progress: {self.report.completed_iterations}/{self.report.total_iterations} iterations",
            f"Status: {self.report.status}",
            f"Total cost: ${self.report.total_cost:.4f}",
        ]

        if self.report.fitness_progression:
            best = max(self.report.fitness_progression)
            latest = self.report.fitness_progression[-1]
            lines.append(f"Best fitness: {best:.3f} (latest: {latest:.3f})")

        return "\n".join(lines)


def load_report(report_path: str) -> OptimizationReport:
    """
    Load a report from file.

    Args:
        report_path: Path to the report JSON file

    Returns:
        OptimizationReport object
    """
    with open(report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    iterations = [
        IterationResult(**it) for it in data.get('iterations', [])
    ]

    top_prompts = [
        PromptResult(**p) for p in data.get('top_prompts', [])
    ]

    return OptimizationReport(
        run_id=data['run_id'],
        start_time=data['start_time'],
        end_time=data.get('end_time'),
        config_summary=data.get('config_summary', {}),
        total_iterations=data['total_iterations'],
        completed_iterations=data['completed_iterations'],
        total_evaluations=data.get('total_evaluations', 0),
        total_cost=data.get('total_cost', 0.0),
        fitness_progression=data.get('fitness_progression', []),
        iterations=iterations,
        top_prompts=top_prompts,
        status=data.get('status', 'unknown')
    )


def load_best_prompt(prompts_dir: str, rank: int = 1) -> Optional[PromptTemplate]:
    """
    Load a best prompt from YAML file.

    Args:
        prompts_dir: Path to best_prompts directory
        rank: Rank of prompt to load (1 = best)

    Returns:
        PromptTemplate or None if not found
    """
    prompt_file = Path(prompts_dir) / f"prompt_{rank:02d}.yaml"

    if not prompt_file.exists():
        return None

    with open(prompt_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    return PromptTemplate(
        system_prompt=data['system_prompt'],
        user_prompt=data['user_prompt'],
        id=data['id'],
        generation=data.get('generation', 0),
        mutation_history=data.get('mutation_history', []),
        fitness=data.get('fitness', 0.0)
    )
