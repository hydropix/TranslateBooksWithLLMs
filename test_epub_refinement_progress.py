"""
Test script to verify EPUB progress tracking with refinement enabled.
"""

from src.core.epub.translation_metrics import TranslationMetrics

def test_epub_refinement_progress():
    """Test that EPUB refinement doubles the total work and progress is calculated correctly."""

    print("=" * 60)
    print("Testing EPUB TranslationMetrics with Refinement")
    print("=" * 60)

    # Create metrics with refinement enabled
    stats = TranslationMetrics()
    stats.total_chunks = 5  # 5 chunks to translate
    stats.enable_refinement = True
    stats.refinement_phase = False

    print(f"\nInitial state:")
    stats_dict = stats.to_dict()
    print(f"  Total chunks: {stats_dict['total_chunks']}")
    print(f"  Completed chunks: {stats_dict['completed_chunks']}")
    print(f"  Progress: {(stats_dict['completed_chunks'] / stats_dict['total_chunks']) * 100:.1f}%")

    # Complete phase 1 (translation)
    print("\n--- PHASE 1: TRANSLATION ---")
    for i in range(5):
        stats.successful_first_try = i + 1
        stats.record_processed()  # Mark chunk as processed
        stats_dict = stats.to_dict()
        progress = (stats_dict['completed_chunks'] / stats_dict['total_chunks']) * 100
        expected = (i + 1) * 10  # 10%, 20%, 30%, 40%, 50%
        print(f"  Chunk {i+1}/5 completed: {progress:.1f}% (should be {expected:.0f}%)")

    stats_dict = stats.to_dict()
    progress = (stats_dict['completed_chunks'] / stats_dict['total_chunks']) * 100
    print(f"\nAfter translation phase:")
    print(f"  Completed chunks: {stats_dict['completed_chunks']}/{stats_dict['total_chunks']}")
    print(f"  Progress: {progress:.1f}% (should be 50.0%)")

    if abs(progress - 50.0) > 0.1:
        print(f"  [ERROR] Expected 50%, got {progress:.1f}%")
        return False
    else:
        print(f"  [OK] Correct!")

    # Start phase 2 (refinement)
    print("\n--- PHASE 2: REFINEMENT ---")
    stats.refinement_phase = True
    stats.refinement_chunks_completed = 0

    stats_dict = stats.to_dict()
    progress = (stats_dict['completed_chunks'] / stats_dict['total_chunks']) * 100
    print(f"After switching to refinement phase:")
    print(f"  Progress: {progress:.1f}% (should still be 50.0%)")

    # Complete phase 2 (refinement)
    for i in range(5):
        stats.refinement_chunks_completed = i + 1
        stats_dict = stats.to_dict()
        progress = (stats_dict['completed_chunks'] / stats_dict['total_chunks']) * 100
        expected = 50.0 + ((i + 1) * 10.0)  # 60%, 70%, 80%, 90%, 100%
        print(f"  Chunk {i+1}/5 refined: {progress:.1f}% (should be {expected:.0f}%)")

    stats_dict = stats.to_dict()
    progress = (stats_dict['completed_chunks'] / stats_dict['total_chunks']) * 100
    print(f"\nFinal state:")
    print(f"  Completed chunks: {stats_dict['completed_chunks']}/{stats_dict['total_chunks']}")
    print(f"  Progress: {progress:.1f}% (should be 100.0%)")

    if abs(progress - 100.0) > 0.1:
        print(f"  [ERROR] Expected 100%, got {progress:.1f}%")
        return False
    else:
        print(f"  [OK] Correct!")

    print("\n" + "=" * 60)
    print("[SUCCESS] All tests passed!")
    print("=" * 60)
    return True


def test_single_phase_epub_progress():
    """Test that single-phase EPUB mode still works correctly."""

    print("\n" + "=" * 60)
    print("Testing EPUB TranslationMetrics WITHOUT Refinement")
    print("=" * 60)

    # Create metrics without refinement
    stats = TranslationMetrics()
    stats.total_chunks = 5
    stats.enable_refinement = False

    print(f"\nInitial state:")
    stats_dict = stats.to_dict()
    print(f"  Total chunks: {stats_dict['total_chunks']}")
    print(f"  Completed chunks: {stats_dict['completed_chunks']}")
    print(f"  Progress: {(stats_dict['completed_chunks'] / stats_dict['total_chunks']) * 100:.1f}%")

    # Complete all chunks
    for i in range(5):
        stats.successful_first_try = i + 1
        stats.record_processed()  # Mark chunk as processed
        stats_dict = stats.to_dict()
        progress = (stats_dict['completed_chunks'] / stats_dict['total_chunks']) * 100
        expected = (i + 1) * 20.0
        print(f"  Chunk {i+1}/5 completed: {progress:.1f}% (should be {expected:.0f}%)")

    stats_dict = stats.to_dict()
    progress = (stats_dict['completed_chunks'] / stats_dict['total_chunks']) * 100
    print(f"\nFinal state:")
    print(f"  Progress: {progress:.1f}% (should be 100.0%)")

    if abs(progress - 100.0) > 0.1:
        print(f"  [ERROR] Expected 100%, got {progress:.1f}%")
        return False
    else:
        print(f"  [OK] Correct!")

    print("\n" + "=" * 60)
    print("[SUCCESS] Single-phase test passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success1 = test_single_phase_epub_progress()
    success2 = test_epub_refinement_progress()

    if success1 and success2:
        print("\n[SUCCESS] All EPUB progress tracking tests passed!")
    else:
        print("\n[FAILED] Some tests failed!")
        exit(1)
