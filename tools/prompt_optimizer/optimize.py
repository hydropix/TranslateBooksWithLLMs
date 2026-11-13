"""
Main optimization script - entry point for prompt optimization.

Uses LLM-based intelligent mutation to improve translation prompts.

Usage:
    python -m tools.prompt_optimizer.optimize --config prompt_optimizer_config.yaml
    python -m tools.prompt_optimizer.optimize --config config.yaml --iterations 20 --population 10
"""

import argparse
import asyncio
import sys
import time
from typing import Optional, Callable

from tools.prompt_optimizer.config import load_config, validate_config, OptimizerConfig
from tools.prompt_optimizer.prompt_template import (
    PromptTemplate, EvaluationFeedback, MutationStrategy,
    create_initial_population, get_mutation_prompt,
    prepare_population_for_evolution
)
from tools.prompt_optimizer.llm_adapter import LLMAdapter, EvaluationResult
from tools.prompt_optimizer.fitness import FitnessCalculator
from tools.prompt_optimizer.cross_validator import CrossValidator, CrossValidationSplit, describe_split
from tools.prompt_optimizer.history import HistoryManager
from tools.prompt_optimizer.logger import get_logger, log_callback_factory


class PromptOptimizer:
    """
    Main prompt optimization engine.

    Implements a genetic algorithm with:
    - Population of prompt templates
    - Fitness evaluation via LLM translation and evaluation
    - LLM-based intelligent mutation (not random rules)
    - Tournament selection
    - Cross-validation to prevent overfitting
    """

    def __init__(
        self,
        config: OptimizerConfig,
        log_callback: Optional[Callable[[str, str], None]] = None,
        verbose: bool = True
    ):
        """
        Initialize the optimizer.

        Args:
            config: Optimizer configuration
            log_callback: Optional logging callback (level, message)
            verbose: Whether to show detailed output
        """
        self.config = config
        self.console = get_logger(verbose=verbose)
        self.log_callback = log_callback or log_callback_factory(verbose)

        # Components
        self.llm = LLMAdapter(config, self.log_callback, self.console)
        self.fitness_calculator = FitnessCalculator(config.fitness)
        self.cross_validator = CrossValidator(config.cross_validation)
        self.history = HistoryManager(config.output_dir)

        # State
        self.population: list[PromptTemplate] = []
        self.cv_split: Optional[CrossValidationSplit] = None
        self.total_evaluations = 0
        self.current_iteration = 0

    async def initialize(self) -> None:
        """Initialize the optimizer and create initial population."""
        self._log("info", "Initializing prompt optimizer...")

        # Create base template
        base_template = PromptTemplate(
            system_prompt=self.config.initial_system_prompt,
            user_prompt=self.config.initial_user_prompt,
            id="base",
            generation=0
        )

        # Create initial population (all copies of base for first iteration)
        self.population = create_initial_population(
            base_template,
            self.config.optimization.population_size
        )
        self._log("info", f"Created initial population of {len(self.population)} prompts")

        # Create cross-validation split
        self.cv_split = self.cross_validator.create_split(self.config.texts)
        self._log("info", describe_split(self.cv_split))

        # Start history tracking
        language_pairs = set()
        for text in self.config.texts:
            language_pairs.add(f"{text.source_language}->{text.target_language}")

        config_summary = {
            'ollama_model': self.config.ollama.model,
            'evaluator_model': self.config.openrouter.model,
            'mutation_model': self.config.openrouter.model,
            'population_size': self.config.optimization.population_size,
            'iterations': self.config.optimization.iterations,
            'k_folds': self.config.cross_validation.k_folds,
            'num_texts': len(self.config.texts),
            'language_pairs': list(language_pairs),
            'mutation_type': 'llm_based'
        }
        self.history.start_run(config_summary, self.config.optimization.iterations)

    async def evaluate_prompt(
        self,
        prompt: PromptTemplate,
        texts: list,
        is_train: bool = True
    ) -> list[EvaluationResult]:
        """
        Evaluate a prompt on a set of texts.

        Args:
            prompt: The prompt to evaluate
            texts: List of TextConfig to evaluate on
            is_train: Whether this is training or test evaluation

        Returns:
            List of evaluation results
        """
        results = []

        for text in texts:
            source_lang = text.source_language
            target_lang = text.target_language

            # Render prompts
            system = prompt.render_system_prompt(source_lang, target_lang)
            user = prompt.render_user_prompt(text.content, source_lang, target_lang)

            # Translate and evaluate
            translation, evaluation = await self.llm.translate_and_evaluate(
                system_prompt=system,
                user_prompt=user,
                source_text=text.content,
                source_language=source_lang,
                target_language=target_lang,
                text_style=text.style,
                text_title=text.title,
                text_author=text.author
            )

            self.total_evaluations += 1

            if evaluation.success:
                score = evaluation.weighted_score
                if is_train:
                    prompt.train_scores.append(score)
                else:
                    prompt.test_scores.append(score)

                # Store feedback for mutation
                feedback = EvaluationFeedback(
                    text_id=text.id,
                    text_title=text.title,
                    source_language=source_lang,
                    target_language=target_lang,
                    score=score,
                    accuracy=evaluation.accuracy,
                    fluency=evaluation.fluency,
                    style=evaluation.style,
                    feedback=evaluation.feedback,
                    translation_excerpt=translation.text[:200] if translation.text else ""
                )
                prompt.evaluation_feedbacks.append(feedback)

            results.append(evaluation)

            # Rate limiting
            await asyncio.sleep(0.3)

        return results

    async def evaluate_population(self, iteration: int) -> None:
        """
        Evaluate the entire population on the current fold.

        Args:
            iteration: Current iteration number
        """
        if not self.cv_split:
            return

        fold = self.cv_split.get_current_fold()
        self._log("info", f"Evaluating population on fold {fold.fold_number}")

        for i, prompt in enumerate(self.population):
            self._log("info", f"  Prompt {i+1}/{len(self.population)} (id={prompt.id})")

            # Clear previous scores and feedbacks
            prompt.train_scores = []
            prompt.test_scores = []
            prompt.evaluation_feedbacks = []

            # Evaluate on train set
            train_results = await self.evaluate_prompt(prompt, fold.train_texts, is_train=True)

            # Evaluate on test set
            test_results = await self.evaluate_prompt(prompt, fold.test_texts, is_train=False)

            # Calculate fitness
            full_prompt = prompt.system_prompt + prompt.user_prompt
            fitness_score = self.fitness_calculator.calculate_fitness(
                train_results, test_results, full_prompt
            )
            prompt.fitness = fitness_score.final_fitness

            self._log("debug", f"    Fitness: {prompt.fitness:.3f} (tokens: ~{prompt.token_estimate})")

    async def mutate_prompt_with_llm(
        self,
        parent: PromptTemplate,
        strategy: MutationStrategy
    ) -> PromptTemplate:
        """
        Create a mutated prompt using the LLM.

        Args:
            parent: The parent template to mutate
            strategy: The mutation strategy to use

        Returns:
            New mutated PromptTemplate
        """
        # Get the mutation prompt
        system_prompt, user_prompt = get_mutation_prompt(
            strategy=strategy,
            template=parent,
            all_templates=self.population
        )

        # Call the LLM with logging context
        new_system_prompt, success, error = await self.llm.mutation.mutate_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            strategy=strategy.value,
            parent_id=parent.id,
            parent_fitness=parent.fitness,
            parent_tokens=parent.token_estimate,
            feedbacks=parent.evaluation_feedbacks
        )

        if not success:
            child = parent.copy()
            child.mutation_history.append(f"{strategy.value}_failed")
            return child

        # Create new template with mutated prompt
        child = PromptTemplate(
            system_prompt=new_system_prompt,
            user_prompt=parent.user_prompt,  # Keep user prompt unchanged
            parent_id=parent.id,
            generation=parent.generation + 1,
            mutation_history=parent.mutation_history + [strategy.value]
        )

        return child

    async def evolve_population(self) -> None:
        """
        Evolve the population using LLM-based mutation.
        """
        self._log("info", "Evolving population with LLM mutations...")

        # Prepare evolution tasks
        elites, mutation_tasks = prepare_population_for_evolution(
            population=self.population,
            elite_count=self.config.optimization.elite_count,
            population_size=self.config.optimization.population_size,
            tournament_size=self.config.optimization.tournament_size
        )

        # Keep elites
        new_population = [elite.copy() for elite in elites]
        self._log("info", f"  Keeping {len(elites)} elites")

        # Perform mutations
        for i, (parent, strategy) in enumerate(mutation_tasks):
            self._log("info", f"  Mutation {i+1}/{len(mutation_tasks)}: {strategy.value}")
            child = await self.mutate_prompt_with_llm(parent, strategy)
            new_population.append(child)

            # Rate limit mutations
            await asyncio.sleep(0.5)

        self.population = new_population
        self._log("info", f"Population evolved to generation {self.population[0].generation}")

    async def run_iteration(self, iteration: int) -> float:
        """
        Run a single optimization iteration.

        Args:
            iteration: Iteration number

        Returns:
            Best fitness score
        """
        start_time = time.time()
        self.current_iteration = iteration

        self.console.header(f"ITERATION {iteration}/{self.config.optimization.iterations}")

        # Evaluate population
        await self.evaluate_population(iteration)

        # Find best
        best_prompt = max(self.population, key=lambda p: p.fitness)
        avg_fitness = sum(p.fitness for p in self.population) / len(self.population)

        # Log fitness for each prompt
        for prompt in sorted(self.population, key=lambda p: p.fitness, reverse=True):
            self.console.fitness_summary(
                prompt_id=prompt.id,
                fitness=prompt.fitness,
                train_scores=prompt.train_scores,
                test_scores=prompt.test_scores
            )

        # Record history
        elapsed = time.time() - start_time
        eval_cost = self.llm.evaluation.total_cost
        mutation_cost = self.llm.mutation.total_cost
        self.history.record_iteration(
            iteration=iteration,
            fold=self.cv_split.get_current_fold().fold_number if self.cv_split else 0,
            population=self.population,
            elapsed_seconds=elapsed,
            evaluation_cost=eval_cost + mutation_cost
        )

        # Iteration summary
        self.console.iteration_summary(
            iteration=iteration,
            total=self.config.optimization.iterations,
            best_fitness=best_prompt.fitness,
            avg_fitness=avg_fitness,
            elapsed=elapsed
        )

        # Evolve population (except on last iteration)
        if iteration < self.config.optimization.iterations:
            await self.evolve_population()

        # Advance fold
        if self.cv_split:
            if not self.cv_split.advance_fold():
                self.cv_split.reset()
                self._log("info", "Cross-validation folds rotated")

        return best_prompt.fitness

    async def run_final_validation(self) -> None:
        """Run final validation on holdout set."""
        if not self.cv_split or not self.cv_split.holdout_text:
            self._log("info", "No holdout text configured, skipping final validation")
            return

        self._log("info", "=== Final Validation on Holdout ===")
        holdout = self.cv_split.holdout_text

        # Evaluate top prompts on holdout
        sorted_pop = sorted(self.population, key=lambda p: p.fitness, reverse=True)

        for i, prompt in enumerate(sorted_pop[:3]):  # Top 3
            self._log("info", f"Validating prompt {prompt.id} (rank {i+1})")

            source_lang = holdout.source_language
            target_lang = holdout.target_language

            system = prompt.render_system_prompt(source_lang, target_lang)
            user = prompt.render_user_prompt(holdout.content, source_lang, target_lang)

            translation, evaluation = await self.llm.translate_and_evaluate(
                system_prompt=system,
                user_prompt=user,
                source_text=holdout.content,
                source_language=source_lang,
                target_language=target_lang,
                text_style=holdout.style,
                text_title=holdout.title,
                text_author=holdout.author
            )

            if evaluation.success:
                self._log("info", f"  Holdout score: {evaluation.weighted_score:.2f}")
                self._log("info", f"  Feedback: {evaluation.feedback}")

    async def run(self) -> None:
        """Run the complete optimization process."""
        try:
            await self.initialize()

            # Main optimization loop
            for iteration in range(1, self.config.optimization.iterations + 1):
                await self.run_iteration(iteration)

            # Final validation
            if self.config.cross_validation.holdout_final:
                await self.run_final_validation()

            # Finalize
            self.history.finalize_run(
                population=self.population,
                total_evaluations=self.total_evaluations,
                status="completed"
            )

            # Print summary
            eval_cost = self.llm.evaluation.get_cost_summary()
            mutation_cost = self.llm.mutation.get_cost_summary()
            total_cost = eval_cost['total_cost_usd'] + mutation_cost['total_cost_usd']

            # Show best prompt
            best = max(self.population, key=lambda p: p.fitness)

            self.console.final_summary(
                best_prompt=best.system_prompt,
                best_fitness=best.fitness,
                total_cost=total_cost,
                mutation_history=best.mutation_history
            )

            self._log("info", f"Results saved to: {self.history.run_dir}")

        except KeyboardInterrupt:
            self._log("warning", "Optimization interrupted by user")
            self.history.finalize_run(
                population=self.population,
                total_evaluations=self.total_evaluations,
                status="interrupted"
            )

        except Exception as e:
            self._log("error", f"Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            self.history.finalize_run(
                population=self.population,
                total_evaluations=self.total_evaluations,
                status="failed"
            )
            raise

        finally:
            await self.llm.close()

    def _log(self, level: str, message: str) -> None:
        """Log a message."""
        self.log_callback(level, message)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prompt Optimizer - Automatically optimize translation prompts using LLM-based mutation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.prompt_optimizer.optimize --config prompt_optimizer_config.yaml
  python -m tools.prompt_optimizer.optimize --config config.yaml --iterations 20
  python -m tools.prompt_optimizer.optimize --config config.yaml --population 10 --verbose
        """
    )

    parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--iterations', '-i',
        type=int,
        help='Override number of iterations (default: from config)'
    )

    parser.add_argument(
        '--population', '-p',
        type=int,
        help='Override population size (default: from config)'
    )

    parser.add_argument(
        '--output', '-o',
        help='Override output directory (default: from config)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate config and show plan without running'
    )

    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1

    # Apply command-line overrides
    if args.iterations:
        config.optimization.iterations = args.iterations
    if args.population:
        config.optimization.population_size = args.population
    if args.output:
        config.output_dir = args.output

    # Validate configuration
    warnings = validate_config(config)
    for warning in warnings:
        print(f"Warning: {warning}")

    if not config.openrouter.api_key:
        print("Error: OpenRouter API key not configured")
        print("Set OPENROUTER_API_KEY in .env or in the config file")
        return 1

    # Dry run mode
    if args.dry_run:
        print("=== Configuration Summary ===")
        print(f"Test LLM: {config.ollama.model} @ {config.ollama.endpoint}")
        print(f"Evaluator/Mutator: {config.openrouter.model}")
        print(f"Texts: {len(config.texts)}")
        print(f"Iterations: {config.optimization.iterations}")
        print(f"Population: {config.optimization.population_size}")
        print(f"Output: {config.output_dir}")
        print(f"\nMutation strategies: CORRECT, SIMPLIFY, REFORMULATE, RADICAL")
        print("(LLM-based intelligent mutation)")
        print("\nConfiguration is valid. Remove --dry-run to start optimization.")
        return 0

    # Run optimization
    optimizer = PromptOptimizer(config, verbose=args.verbose)
    await optimizer.run()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
