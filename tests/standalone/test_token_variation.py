"""
Test script to analyze token estimation variations with qwen3:8b
Compares behavior WITH and WITHOUT runtime thinking model detection
"""
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.core.chunking.token_chunker import TokenChunker
from src.core.context_optimizer import (
    estimate_tokens_with_margin,
    adjust_parameters_for_context,
    STANDARD_CONTEXT_SIZES
)
from src.config import MAX_TOKENS_PER_CHUNK, API_ENDPOINT
from src.core.llm.factory import create_llm_provider
from src.prompts.prompts import generate_translation_prompt


async def analyze_token_variations(input_file: str):
    """Analyze token estimation for each chunk and show why context varies."""

    print("=" * 80)
    print("TOKEN VARIATION ANALYSIS - WITH RUNTIME DETECTION FIX")
    print("=" * 80)
    print(f"\nInput file: {input_file}")
    print(f"Model: qwen3:8b")
    print(f"Base OLLAMA_NUM_CTX: 2048")
    print(f"MAX_TOKENS_PER_CHUNK: {MAX_TOKENS_PER_CHUNK}")
    print(f"Standard context sizes: {STANDARD_CONTEXT_SIZES[:5]}...")
    print()

    # Step 1: Detect thinking model status via runtime test
    print("=" * 80)
    print("STEP 1: RUNTIME THINKING MODEL DETECTION")
    print("=" * 80)

    llm_client = create_llm_provider(
        "ollama",
        api_endpoint=API_ENDPOINT,
        model="qwen3:8b",
        context_window=2048
    )

    detected_thinking = await llm_client.detect_thinking_model()
    print(f"\nDetected is_thinking_model: {detected_thinking}")

    await llm_client.close()

    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"\nFile size: {len(content)} characters")

    # Chunk the content
    chunker = TokenChunker(max_tokens=MAX_TOKENS_PER_CHUNK)
    chunks = chunker.chunk_text(content)

    print(f"Total chunks: {len(chunks)}")
    print()

    # Compare OLD vs NEW behavior
    print("=" * 80)
    print("COMPARISON: OLD (model name check) vs NEW (runtime detection)")
    print("=" * 80)

    base_context_window = 2048
    source_language = "English"
    target_language = "Chinese"
    last_context = ""

    old_distribution = {}
    new_distribution = {}

    for i, chunk_data in enumerate(chunks):
        main_content = chunk_data["main_content"]
        # Generate the prompt
        prompt_pair = generate_translation_prompt(
            main_content,
            last_context,
            source_language,
            target_language
        )

        combined_prompt = prompt_pair.system + "\n\n" + prompt_pair.user

        estimation = estimate_tokens_with_margin(
            text=combined_prompt,
            language=source_language,
            apply_margin=True
        )

        # OLD behavior: uses model name check (is_thinking_model=None -> fallback to THINKING_MODELS list)
        old_num_ctx, _, _ = adjust_parameters_for_context(
            estimated_tokens=estimation.estimated_tokens,
            current_num_ctx=base_context_window,
            current_chunk_size=25,
            model_name="qwen3:8b",
            min_chunk_size=5,
            is_thinking_model=None  # Falls back to model name check
        )

        # NEW behavior: uses runtime detection result
        new_num_ctx, _, _ = adjust_parameters_for_context(
            estimated_tokens=estimation.estimated_tokens,
            current_num_ctx=base_context_window,
            current_chunk_size=25,
            model_name="qwen3:8b",
            min_chunk_size=5,
            is_thinking_model=detected_thinking  # Uses runtime detection
        )

        # Track distributions
        if old_num_ctx not in old_distribution:
            old_distribution[old_num_ctx] = 0
        old_distribution[old_num_ctx] += 1

        if new_num_ctx not in new_distribution:
            new_distribution[new_num_ctx] = 0
        new_distribution[new_num_ctx] += 1

        print(f"Chunk {i+1}: estimated={estimation.estimated_tokens} tokens | "
              f"OLD={old_num_ctx} | NEW={new_num_ctx}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nOLD behavior (model name 'qwen3' -> assumes thinking):")
    for ctx_size in sorted(old_distribution.keys()):
        print(f"  {ctx_size} tokens: {old_distribution[ctx_size]} chunks")

    print(f"\nNEW behavior (runtime detection -> is_thinking={detected_thinking}):")
    for ctx_size in sorted(new_distribution.keys()):
        print(f"  {ctx_size} tokens: {new_distribution[ctx_size]} chunks")

    print("\n" + "=" * 80)
    print("EXPLANATION")
    print("=" * 80)

    if detected_thinking == False:
        print("\nThe model qwen3:8b was detected as NON-THINKING at runtime.")
        print("This means it does NOT produce <think> blocks when think:false is set.")
        print()
        print("OLD behavior incorrectly added +2000 token buffer because 'qwen3' is in THINKING_MODELS list.")
        print("NEW behavior correctly skips the buffer based on actual runtime detection.")
        print()
        print("This explains the jump from 4096/8192 (OLD) to lower values (NEW).")
    elif detected_thinking == True:
        print("\nThe model was detected as THINKING - the +2000 buffer is correctly applied.")
    else:
        print("\nThinking detection returned None - falling back to model name check.")


if __name__ == "__main__":
    input_file = r"C:\Users\Bruno\Documents\GitHub\TestFileToTranslate\book_to_translate_en.txt"

    if not os.path.exists(input_file):
        print(f"ERROR: File not found: {input_file}")
        sys.exit(1)

    asyncio.run(analyze_token_variations(input_file))
