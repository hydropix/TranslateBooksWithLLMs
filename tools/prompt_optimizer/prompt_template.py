"""
Prompt template representation and LLM-based mutation strategies.

The mutation system uses a frontier LLM (Claude) to intelligently improve prompts
based on evaluation feedback, while maintaining generalization and efficiency.
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class MutationStrategy(Enum):
    """
    LLM-based mutation strategies.

    Each strategy explores a different direction for prompt improvement:
    - CORRECT: Fix identified weaknesses from evaluation feedback
    - SIMPLIFY: Remove unnecessary instructions, reduce token cost
    - REFORMULATE: Say the same thing more effectively
    - RADICAL: Try a completely different approach
    """
    CORRECT = "correct"
    SIMPLIFY = "simplify"
    REFORMULATE = "reformulate"
    RADICAL = "radical"


@dataclass
class EvaluationFeedback:
    """Aggregated feedback from evaluations."""
    text_id: str
    text_title: str
    source_language: str
    target_language: str
    score: float
    accuracy: float
    fluency: float
    style: float
    feedback: str
    translation_excerpt: str = ""


@dataclass
class PromptTemplate:
    """
    Represents a prompt template with system and user components.
    """
    system_prompt: str
    user_prompt: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_id: Optional[str] = None
    generation: int = 0
    mutation_history: list[str] = field(default_factory=list)

    # Evaluation results (filled during optimization)
    train_scores: list[float] = field(default_factory=list)
    test_scores: list[float] = field(default_factory=list)
    fitness: float = 0.0

    # Feedback from evaluations (used for intelligent mutation)
    evaluation_feedbacks: list[EvaluationFeedback] = field(default_factory=list)

    def copy(self) -> 'PromptTemplate':
        """Create a deep copy of this template."""
        return PromptTemplate(
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            id=uuid.uuid4().hex[:8],
            parent_id=self.id,
            generation=self.generation + 1,
            mutation_history=self.mutation_history.copy(),
            evaluation_feedbacks=[]  # Fresh start for new template
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'parent_id': self.parent_id,
            'generation': self.generation,
            'system_prompt': self.system_prompt,
            'user_prompt': self.user_prompt,
            'mutation_history': self.mutation_history,
            'train_scores': self.train_scores,
            'test_scores': self.test_scores,
            'fitness': self.fitness
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PromptTemplate':
        """Create from dictionary."""
        return cls(
            system_prompt=data['system_prompt'],
            user_prompt=data['user_prompt'],
            id=data.get('id', uuid.uuid4().hex[:8]),
            parent_id=data.get('parent_id'),
            generation=data.get('generation', 0),
            mutation_history=data.get('mutation_history', []),
            train_scores=data.get('train_scores', []),
            test_scores=data.get('test_scores', []),
            fitness=data.get('fitness', 0.0)
        )

    def render_user_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        """Render the user prompt with the given text and languages."""
        return self.user_prompt.format(
            text=text,
            source_language=source_lang,
            target_language=target_lang
        )

    def render_system_prompt(self, source_lang: str, target_lang: str) -> str:
        """Render the system prompt with the given languages."""
        return self.system_prompt.format(
            source_language=source_lang,
            target_language=target_lang
        )

    @property
    def total_length(self) -> int:
        """Total character count of both prompts."""
        return len(self.system_prompt) + len(self.user_prompt)

    @property
    def token_estimate(self) -> int:
        """Rough token estimate (chars / 4)."""
        return self.total_length // 4

    def get_weakness_summary(self) -> str:
        """Summarize weaknesses from evaluation feedbacks."""
        if not self.evaluation_feedbacks:
            return "No evaluation feedback available."

        lines = []
        for fb in self.evaluation_feedbacks:
            lines.append(f"- {fb.text_title} ({fb.source_language}→{fb.target_language}): "
                        f"score={fb.score:.1f}, accuracy={fb.accuracy:.1f}, "
                        f"fluency={fb.fluency:.1f}, style={fb.style:.1f}")
            if fb.feedback:
                lines.append(f"  Feedback: {fb.feedback[:200]}")

        return "\n".join(lines)


# =============================================================================
# LLM-Based Mutation Prompts
# =============================================================================

MUTATION_SYSTEM_PROMPT = """You are an expert prompt engineer specializing in optimizing translation prompts.

Your task is to improve a translation system prompt based on evaluation feedback.

CRITICAL RULES:
1. The prompt must remain GENERIC - no references to specific texts, authors, or content
2. The prompt must work for ANY source/target language pair
3. Keep placeholders: {source_language}, {target_language}
4. Focus on instructions that help a small LLM (4B parameters) translate better
5. Every instruction must earn its place - remove fluff and redundancy
6. Be concise - longer prompts cost more and aren't always better

The translation LLM is small (4B params) so instructions must be:
- Clear and direct
- Not too complex or nuanced
- Focused on common error patterns"""


def build_correction_prompt(template: PromptTemplate) -> str:
    """Build prompt for CORRECT strategy - fix identified weaknesses."""
    return f"""STRATEGY: CORRECT
Fix the identified weaknesses while keeping the prompt generic and efficient.

CURRENT PROMPT ({template.token_estimate} tokens estimated):
---
{template.system_prompt}
---

EVALUATION FEEDBACK (weaknesses to address):
{template.get_weakness_summary()}

TASK:
Modify the prompt to address the identified weaknesses.
- Add specific instructions to fix recurring problems
- Don't add instructions for problems that didn't occur
- Keep it generic (no text-specific fixes)
- Aim to IMPROVE scores, especially low ones (accuracy, fluency, style)

Return ONLY the improved system prompt, nothing else."""


def build_simplify_prompt(template: PromptTemplate) -> str:
    """Build prompt for SIMPLIFY strategy - remove unnecessary instructions."""
    return f"""STRATEGY: SIMPLIFY
Remove unnecessary instructions to make the prompt more efficient.

CURRENT PROMPT ({template.token_estimate} tokens estimated):
---
{template.system_prompt}
---

EVALUATION SCORES:
{template.get_weakness_summary()}

TASK:
Simplify the prompt by removing:
- Redundant instructions (saying the same thing twice)
- Vague instructions that don't help a small LLM
- Instructions that aren't reflected in the scores
- Unnecessary formatting or verbosity

Goal: Achieve similar or better results with fewer tokens.
A lean prompt is often more effective than a long one.

Return ONLY the simplified system prompt, nothing else."""


def build_reformulate_prompt(template: PromptTemplate) -> str:
    """Build prompt for REFORMULATE strategy - say it better."""
    return f"""STRATEGY: REFORMULATE
Rewrite the prompt to be clearer and more effective, same length or shorter.

CURRENT PROMPT ({template.token_estimate} tokens estimated):
---
{template.system_prompt}
---

EVALUATION SCORES:
{template.get_weakness_summary()}

TASK:
Reformulate the prompt to be more effective:
- Use clearer, more direct language
- Restructure for better flow
- Use formatting (headers, bullets) strategically
- Make instructions more actionable for a small LLM

Keep the same general approach but express it better.

Return ONLY the reformulated system prompt, nothing else."""


def build_radical_prompt(template: PromptTemplate, all_templates: list['PromptTemplate'] = None) -> str:
    """Build prompt for RADICAL strategy - try something different."""

    # Show what approaches have been tried
    tried_approaches = ""
    if all_templates:
        tried_approaches = "\nAPPROACHES ALREADY TRIED:\n"
        for t in all_templates[:3]:  # Show top 3
            excerpt = t.system_prompt[:150].replace('\n', ' ')
            tried_approaches += f"- (fitness={t.fitness:.2f}) {excerpt}...\n"

    return f"""STRATEGY: RADICAL
Try a completely different approach to the translation prompt.

CURRENT BEST PROMPT ({template.token_estimate} tokens, fitness={template.fitness:.2f}):
---
{template.system_prompt}
---

EVALUATION FEEDBACK:
{template.get_weakness_summary()}
{tried_approaches}

TASK:
Create a DIFFERENT prompt approach. Ideas to explore:
- Minimalist: Just the essential instructions
- Role-focused: Strong persona/character for the translator
- Rule-based: Clear numbered rules
- Example-driven: Include a translation example pattern
- Constraint-focused: What NOT to do

Don't just tweak - try something structurally different.
The goal is exploration, not incremental improvement.

Return ONLY the new system prompt, nothing else."""


def get_mutation_prompt(
    strategy: MutationStrategy,
    template: PromptTemplate,
    all_templates: list['PromptTemplate'] = None
) -> tuple[str, str]:
    """
    Get the system and user prompts for a mutation request.

    Returns:
        Tuple of (system_prompt, user_prompt) for the mutation LLM call
    """
    if strategy == MutationStrategy.CORRECT:
        user_prompt = build_correction_prompt(template)
    elif strategy == MutationStrategy.SIMPLIFY:
        user_prompt = build_simplify_prompt(template)
    elif strategy == MutationStrategy.REFORMULATE:
        user_prompt = build_reformulate_prompt(template)
    elif strategy == MutationStrategy.RADICAL:
        user_prompt = build_radical_prompt(template, all_templates)
    else:
        user_prompt = build_correction_prompt(template)

    return MUTATION_SYSTEM_PROMPT, user_prompt


# =============================================================================
# Population Management (no longer needs PromptMutator class)
# =============================================================================

def create_initial_population(
    base_template: PromptTemplate,
    population_size: int
) -> list[PromptTemplate]:
    """
    Create an initial population with just the base template.

    Unlike before, we don't pre-mutate - we let the LLM mutator
    create variations after the first evaluation round.
    """
    population = [base_template]

    # Fill with copies of base (will be mutated after first eval)
    while len(population) < population_size:
        copy = base_template.copy()
        copy.id = uuid.uuid4().hex[:8]
        copy.parent_id = base_template.id
        population.append(copy)

    return population


def tournament_selection(
    population: list[PromptTemplate],
    tournament_size: int
) -> PromptTemplate:
    """Select a template using tournament selection."""
    import random
    contestants = random.sample(population, min(tournament_size, len(population)))
    return max(contestants, key=lambda t: t.fitness)


def select_mutation_strategy(template: PromptTemplate, generation: int) -> MutationStrategy:
    """
    Select a mutation strategy based on context.

    Strategy selection logic:
    - Early generations: More RADICAL exploration
    - Low scores: More CORRECT to fix issues
    - High token count: More SIMPLIFY to reduce cost
    - Otherwise: Mix of strategies
    """
    import random

    # If prompt is long, bias towards simplify
    if template.token_estimate > 300:
        weights = [0.2, 0.5, 0.2, 0.1]  # correct, simplify, reformulate, radical
    # If scores are low, bias towards correct
    elif template.fitness < 6.0:
        weights = [0.5, 0.1, 0.2, 0.2]
    # Early generations, explore more
    elif generation < 3:
        weights = [0.2, 0.1, 0.2, 0.5]
    # Otherwise balanced
    else:
        weights = [0.3, 0.2, 0.3, 0.2]

    strategies = list(MutationStrategy)
    return random.choices(strategies, weights=weights)[0]


def prepare_population_for_evolution(
    population: list[PromptTemplate],
    elite_count: int,
    population_size: int,
    tournament_size: int = 3
) -> list[tuple[PromptTemplate, MutationStrategy]]:
    """
    Prepare the population for evolution by selecting parents and strategies.

    Returns:
        List of (parent_template, mutation_strategy) tuples for non-elite slots
    """

    # Sort by fitness
    sorted_pop = sorted(population, key=lambda t: t.fitness, reverse=True)

    # Elites are kept as-is (handled separately)
    elites = sorted_pop[:elite_count]

    # Prepare mutations for remaining slots
    mutations_needed = population_size - elite_count
    mutation_tasks = []

    for i in range(mutations_needed):
        # Select parent via tournament
        parent = tournament_selection(population, tournament_size)

        # Select strategy
        strategy = select_mutation_strategy(parent, parent.generation)

        mutation_tasks.append((parent, strategy))

    return elites, mutation_tasks
