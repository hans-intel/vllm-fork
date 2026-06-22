#!/usr/bin/env python3
"""Validate the dist-sampling port in ww25.

This script checks:
1. All env vars are properly registered
2. All functions can be imported
3. Function signatures match expected patterns
4. No syntax errors in key modules
"""

import sys
import inspect


def check_env_vars():
    """Check that env vars are registered in vllm.envs."""
    print("Checking environment variables...")
    try:
        import vllm.envs as envs
    except ImportError as e:
        print(f"  ✗ Failed to import vllm.envs: {e}")
        return False

    required_vars = [
        'VLLM_XPU_DIST_SAMPLE',
        'VLLM_XPU_DIST_SAMPLE_FUSED',
        'VLLM_XPU_DIST_SAMPLE_ALLGATHER',
    ]

    all_ok = True
    for var in required_vars:
        if hasattr(envs, var):
            try:
                # Try to access the value (will be False by default)
                val = getattr(envs, var)
                print(f"  ✓ {var} = {val}")
            except Exception as e:
                print(f"  ⚠ {var} defined but failed to access: {e}")
                all_ok = False
        else:
            print(f"  ✗ {var} NOT FOUND")
            all_ok = False

    return all_ok


def check_gumbel_kernel():
    """Check gumbel.py functions."""
    print("\nChecking gumbel.py functions...")
    try:
        from vllm.v1.worker.gpu.sample.gumbel import dist_gumbel_local_packed
    except ImportError as e:
        print(f"  ✗ Failed to import dist_gumbel_local_packed: {e}")
        return False

    print(f"  ✓ dist_gumbel_local_packed imported")

    # Check it's callable
    if not callable(dist_gumbel_local_packed):
        print(f"  ✗ dist_gumbel_local_packed is not callable")
        return False
    print(f"  ✓ dist_gumbel_local_packed is callable")

    # Check signature
    sig = inspect.signature(dist_gumbel_local_packed)
    params = list(sig.parameters.keys())
    expected_params = ['logits', 'seed', 'vocab_start', 'total_vocab']
    if params == expected_params:
        print(f"  ✓ Signature matches: {params}")
    else:
        print(f"  ⚠ Signature mismatch:")
        print(f"    Expected: {expected_params}")
        print(f"    Got:      {params}")
        return False

    return True


def check_logits_processor():
    """Check logits_processor.py methods."""
    print("\nChecking logits_processor.py methods...")
    try:
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
    except ImportError as e:
        print(f"  ✗ Failed to import LogitsProcessor: {e}")
        return False

    required_methods = [
        'gumbel_argmax_tokens',
        '_gumbel_argmax_tokens_fused',
        '_gumbel_argmax_tokens_proto',
    ]

    all_ok = True
    for method_name in required_methods:
        if hasattr(LogitsProcessor, method_name):
            method = getattr(LogitsProcessor, method_name)
            if callable(method):
                print(f"  ✓ LogitsProcessor.{method_name}")
            else:
                print(f"  ✗ LogitsProcessor.{method_name} not callable")
                all_ok = False
        else:
            print(f"  ✗ LogitsProcessor.{method_name} NOT FOUND")
            all_ok = False

    return all_ok


def check_gpu_model_runner():
    """Check gpu_model_runner.py methods."""
    print("\nChecking gpu_model_runner.py methods...")
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError as e:
        print(f"  ✗ Failed to import GPUModelRunner: {e}")
        return False

    if hasattr(GPUModelRunner, '_dist_sample_eligible'):
        method = getattr(GPUModelRunner, '_dist_sample_eligible')
        if callable(method):
            print(f"  ✓ GPUModelRunner._dist_sample_eligible")
        else:
            print(f"  ✗ GPUModelRunner._dist_sample_eligible not callable")
            return False
    else:
        print(f"  ✗ GPUModelRunner._dist_sample_eligible NOT FOUND")
        return False

    return True


def main():
    """Run all validation checks."""
    print("=" * 70)
    print("Dist-Sampling ww25 Port Validation")
    print("=" * 70)
    print("")

    checks = [
        ("Environment Variables", check_env_vars),
        ("Gumbel Kernel", check_gumbel_kernel),
        ("Logits Processor", check_logits_processor),
        ("GPU Model Runner", check_gpu_model_runner),
    ]

    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n✗ {check_name} check failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((check_name, False))

    # Summary
    print("")
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check_name}")

    print(f"\nTotal: {passed}/{total} checks passed")

    if passed == total:
        print("\n✓ All validation checks passed!")
        print("")
        print("Next steps:")
        print("  1. Set environment: export VLLM_XPU_DIST_SAMPLE=1")
        print("  2. Run tests: ./run_ww25_tests.sh 220 mixed_subset220.pkl")
        print("  3. Compare results: python3 measure_latency.py <logdir1> <logdir2> <logdir3>")
        return 0
    else:
        print(f"\n✗ {total - passed} check(s) failed")
        print("Port may have issues. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
