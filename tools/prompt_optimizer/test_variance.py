"""
Test de variance: execute le meme prompt sur le meme texte N fois
pour mesurer la variance naturelle du systeme (Ollama + evaluateur).

Usage:
    python -m tools.prompt_optimizer.test_variance --config prompt_optimizer_config.yaml --runs 5
"""

import argparse
import asyncio
import statistics
import sys

from tools.prompt_optimizer.config import load_config
from tools.prompt_optimizer.llm_adapter import LLMAdapter
from tools.prompt_optimizer.prompt_template import PromptTemplate


async def run_variance_test(config_path: str, num_runs: int = 5):
    """Execute le test de variance."""

    print(f"Chargement de la configuration: {config_path}")
    config = load_config(config_path)

    if not config.openrouter.api_key:
        print("Erreur: OPENROUTER_API_KEY non configure")
        return 1

    # Utiliser le premier texte et le prompt de base
    text = config.texts[0]
    print(f"\nTexte de test: {text.title} ({text.source_language} -> {text.target_language})")
    print(f"Extrait: {text.content[:100]}...")

    # Creer le prompt de base
    prompt = PromptTemplate(
        system_prompt=config.initial_system_prompt,
        user_prompt=config.initial_user_prompt,
        id="base",
        generation=0
    )

    print(f"\nPrompt system (debut): {prompt.system_prompt[:150]}...")
    print(f"\nExecution de {num_runs} tests identiques...")
    print("=" * 60)

    # Adapter LLM
    def log(level, msg):
        if level == "error":
            print(f"[ERREUR] {msg}")

    llm = LLMAdapter(config, log)

    results = []
    translations = []

    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")

        system = prompt.render_system_prompt(text.source_language, text.target_language)
        user = prompt.render_user_prompt(text.content, text.source_language, text.target_language)

        translation, evaluation = await llm.translate_and_evaluate(
            system_prompt=system,
            user_prompt=user,
            source_text=text.content,
            source_language=text.source_language,
            target_language=text.target_language,
            text_style=text.style,
            text_title=text.title,
            text_author=text.author
        )

        if evaluation.success:
            score = evaluation.weighted_score
            results.append(score)
            translations.append(translation[:100] if translation else "N/A")

            print(f"Score: {score:.2f}")
            print(f"  Accuracy: {evaluation.accuracy:.1f}")
            print(f"  Fluency:  {evaluation.fluency:.1f}")
            print(f"  Style:    {evaluation.style:.1f}")
            print(f"  Overall:  {evaluation.overall:.1f}")
            print(f"  Traduction: {translation[:80]}...")
        else:
            print(f"ECHEC: {evaluation.feedback}")

        # Petit delai entre les runs
        await asyncio.sleep(1)

    await llm.close()

    # Analyse
    print("\n" + "=" * 60)
    print("RESULTATS")
    print("=" * 60)

    if len(results) >= 2:
        mean = statistics.mean(results)
        stdev = statistics.stdev(results)
        min_score = min(results)
        max_score = max(results)

        print(f"\nScores: {[f'{r:.2f}' for r in results]}")
        print(f"\nMoyenne:     {mean:.3f}")
        print(f"Ecart-type:  {stdev:.3f}")
        print(f"Min:         {min_score:.2f}")
        print(f"Max:         {max_score:.2f}")
        print(f"Plage:       {max_score - min_score:.2f}")

        print("\n--- INTERPRETATION ---")
        if stdev < 0.3:
            print("Variance FAIBLE (<0.3) - Le systeme est stable.")
            print("Les differences entre mutations sont significatives.")
        elif stdev < 0.6:
            print("Variance MOYENNE (0.3-0.6) - Attention aux faux positifs.")
            print("Une amelioration de <0.5 pourrait etre du bruit.")
        else:
            print("Variance ELEVEE (>0.6) - Le systeme est instable!")
            print("Les scores d'optimisation ne sont pas fiables.")

        # Verifier si les traductions sont identiques
        print("\n--- TRADUCTIONS ---")
        unique_translations = len(set(translations))
        if unique_translations == 1:
            print("Toutes les traductions sont IDENTIQUES")
            print("-> La variance vient uniquement de l'evaluateur")
        else:
            print(f"{unique_translations} traductions differentes sur {num_runs}")
            print("-> Ollama genere des reponses differentes (temperature > 0?)")
    else:
        print("Pas assez de resultats pour l'analyse")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Test de variance du systeme")
    parser.add_argument('--config', '-c', required=True, help='Fichier de configuration YAML')
    parser.add_argument('--runs', '-r', type=int, default=5, help='Nombre de runs (defaut: 5)')

    args = parser.parse_args()

    return asyncio.run(run_variance_test(args.config, args.runs))


if __name__ == "__main__":
    sys.exit(main())
