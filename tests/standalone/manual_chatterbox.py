#!/usr/bin/env python3
"""
Script de test complet pour Chatterbox TTS
Vérifie l'installation et teste la génération audio
"""

import sys
import os

def print_section(title):
    """Affiche un titre de section"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_imports():
    """Teste l'importation des modules nécessaires"""
    print_section("Test des imports")

    modules = {
        'torch': 'PyTorch',
        'torchaudio': 'TorchAudio',
        'chatterbox': 'Chatterbox TTS (module)'
    }

    all_ok = True
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"✓ {name}: OK")
        except ImportError as e:
            print(f"✗ {name}: ERREUR - {e}")
            all_ok = False

    # Test import spécifique de ChatterboxTTS
    try:
        from chatterbox import ChatterboxTTS  # noqa: F401 - import check
        print(f"✓ ChatterboxTTS class: OK")
    except ImportError as e:
        print(f"✗ ChatterboxTTS class: ERREUR - {e}")
        all_ok = False

    return all_ok

def test_pytorch_cuda():
    """Teste PyTorch et CUDA"""
    print_section("Test de PyTorch et CUDA")

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"Version CUDA compilée: {torch.version.cuda if torch.version.cuda else 'CPU seulement'}")
        print(f"CUDA disponible: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"Nombre de GPU: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    VRAM totale: {props.total_memory / 1024**3:.2f} GB")
                print(f"    VRAM libre: {(props.total_memory - torch.cuda.memory_allocated(i)) / 1024**3:.2f} GB")
                print(f"    Capacité de calcul: {props.major}.{props.minor}")
            return True, "cuda"
        else:
            print("\n⚠ ATTENTION: CUDA n'est pas disponible.")
            print("Le mode CPU sera utilisé (plus lent).")
            print("\nPour activer le GPU:")
            print("  1. Vérifiez que vous avez un GPU NVIDIA")
            print("  2. Installez les pilotes NVIDIA récents")
            print("  3. Exécutez: fix_pytorch_cuda.bat")
            return True, "cpu"

    except Exception as e:
        print(f"✗ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_chatterbox_init(device):
    """Teste l'initialisation de Chatterbox TTS"""
    print_section(f"Test d'initialisation de Chatterbox TTS ({device.upper()})")

    try:
        from chatterbox import ChatterboxTTS
        print(f"Initialisation sur le périphérique: {device}")

        # Mesurer le temps d'initialisation
        import time
        start = time.time()
        tts = ChatterboxTTS(device=device)
        init_time = time.time() - start

        print(f"✓ Chatterbox TTS initialisé en {init_time:.2f} secondes")
        return True, tts

    except Exception as e:
        print(f"✗ ERREUR lors de l'initialisation: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_audio_generation(tts, device):
    """Teste la génération audio"""
    print_section("Test de génération audio")

    try:
        import time
        import numpy as np

        test_text = "Hello! This is a test of Chatterbox text to speech synthesis."
        print(f"Texte de test: \"{test_text}\"")
        print(f"Périphérique: {device.upper()}")

        # Test de génération
        print("\nGénération en cours...")
        start = time.time()
        audio = tts.tts(test_text)  # ChatterboxTTS utilise .tts() et non .synthesize()
        gen_time = time.time() - start

        # Vérifier le résultat
        if audio is None or len(audio) == 0:
            print("✗ ERREUR: Aucun audio généré")
            return False

        # Statistiques
        audio_array = np.array(audio)
        duration = len(audio_array) / 24000  # Chatterbox utilise 24kHz

        print(f"\n✓ Audio généré avec succès!")
        print(f"  Échantillons: {len(audio_array):,}")
        print(f"  Durée audio: {duration:.2f} secondes")
        print(f"  Temps de génération: {gen_time:.2f} secondes")
        print(f"  Vitesse: {duration/gen_time:.2f}x temps réel")
        print(f"  Amplitude min/max: {audio_array.min():.4f} / {audio_array.max():.4f}")

        # Sauvegarder l'audio de test
        try:
            import soundfile as sf
            output_file = "test_chatterbox_output.wav"
            sf.write(output_file, audio_array, 24000)
            print(f"\n✓ Audio sauvegardé: {output_file}")
            print(f"  Taille du fichier: {os.path.getsize(output_file) / 1024:.1f} KB")
        except Exception as e:
            print(f"\n⚠ Impossible de sauvegarder l'audio: {e}")
            print("  Installez soundfile: pip install soundfile")

        return True

    except Exception as e:
        print(f"✗ ERREUR lors de la génération: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """Compare les performances CPU vs GPU si disponible"""
    print_section("Test de performance comparatif")

    try:
        import torch
        from chatterbox import ChatterboxTTS
        import time

        test_text = "Performance test for text to speech synthesis."
        results = {}

        # Test CPU
        print("Test CPU...")
        try:
            tts_cpu = ChatterboxTTS(device="cpu")
            start = time.time()
            tts_cpu.tts(test_text)
            cpu_time = time.time() - start
            results['cpu'] = cpu_time
            print(f"  Temps CPU: {cpu_time:.2f}s")
        except Exception as e:
            print(f"  Erreur CPU: {e}")

        # Test GPU si disponible
        if torch.cuda.is_available():
            print("Test GPU...")
            try:
                tts_gpu = ChatterboxTTS(device="cuda")
                # Warm-up
                _ = tts_gpu.tts("warmup")

                start = time.time()
                tts_gpu.tts(test_text)
                gpu_time = time.time() - start
                results['gpu'] = gpu_time
                print(f"  Temps GPU: {gpu_time:.2f}s")

                if 'cpu' in results and 'gpu' in results:
                    speedup = results['cpu'] / results['gpu']
                    print(f"\n✓ Accélération GPU: {speedup:.1f}x plus rapide que CPU")
            except Exception as e:
                print(f"  Erreur GPU: {e}")
        else:
            print("GPU non disponible, comparaison impossible")

        return True

    except Exception as e:
        print(f"Erreur lors du test de performance: {e}")
        return False

def main():
    """Fonction principale"""
    print("\n" + "=" * 70)
    print("  TEST COMPLET DE CHATTERBOX TTS")
    print("=" * 70)
    print("\nCe script va tester votre installation de Chatterbox TTS")
    print("et vérifier que tout fonctionne correctement.\n")

    # 1. Test des imports
    if not test_imports():
        print("\n❌ ÉCHEC: Certains modules ne sont pas installés.")
        print("Exécutez: install_chatterbox.bat")
        return 1

    # 2. Test PyTorch et CUDA
    pytorch_ok, device = test_pytorch_cuda()
    if not pytorch_ok:
        print("\n❌ ÉCHEC: Problème avec PyTorch.")
        return 1

    # 3. Test initialisation
    init_ok, tts = test_chatterbox_init(device)
    if not init_ok:
        print("\n❌ ÉCHEC: Impossible d'initialiser Chatterbox TTS.")
        return 1

    # 4. Test génération audio
    gen_ok = test_audio_generation(tts, device)
    if not gen_ok:
        print("\n❌ ÉCHEC: Impossible de générer de l'audio.")
        return 1

    # 5. Test de performance (optionnel)
    print("\n")
    response = input("Effectuer un test de performance CPU vs GPU? (o/N): ")
    if response.lower() == 'o':
        test_performance_comparison()

    # Résumé final
    print_section("RÉSUMÉ")
    print("✓ Tous les tests sont passés avec succès!")
    print(f"\nConfiguration:")
    print(f"  Périphérique utilisé: {device.upper()}")

    if device == "cpu":
        print("\n⚠ Mode CPU détecté")
        print("  La génération audio sera lente.")
        print("  Pour de meilleures performances:")
        print("    1. Vérifiez que vous avez un GPU NVIDIA")
        print("    2. Exécutez: diagnose_gpu.bat")
        print("    3. Suivez les recommandations pour activer CUDA")
    else:
        print("\n✓ Mode GPU activé")
        print("  La génération audio sera rapide.")
        print("  Chatterbox TTS peut générer de l'audio en temps réel.")

    print("\n🎉 Chatterbox TTS est prêt à être utilisé!")
    print("\nPour l'utiliser dans l'application:")
    print("  1. Lancez: start.bat")
    print("  2. Dans l'interface web, sélectionnez 'Chatterbox TTS'")
    print("  3. Uploadez optionnellement un échantillon vocal pour le clonage")

    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        print("\n" + "=" * 70)
        input("\nAppuyez sur Entrée pour quitter...")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrompu par l'utilisateur.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()
        input("\nAppuyez sur Entrée pour quitter...")
        sys.exit(1)
