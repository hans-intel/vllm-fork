#!/usr/bin/env python3
"""Verify dist-sample's shard-local MinTokens masking == stock full-vocab masking.

The stock MinTokensLogitsProcessor sets logits[row, eos_id] = -inf over the full
[B, V] tensor. Dist-sample shards the vocab across TP ranks and masks only the
ids that land in each rank's [vocab_start, vocab_start+width) slice. This test
asserts: for every (row, token) the stock path censors, exactly one shard censors
it, and no shard censors anything else.

Pure CPU/torch; no XPU needed. Run anywhere.
"""
import sys
import torch


def _apply_min_tokens_mask(logits, vocab_start, min_tokens_mask):
    """Copy of LogitsProcessor._apply_min_tokens_mask (kept in sync for testing)."""
    if min_tokens_mask is None:
        return
    rows, global_tok_ids = min_tokens_mask
    if rows.numel() == 0:
        return
    local_cols = global_tok_ids.to(torch.int64) - vocab_start
    in_shard = (local_cols >= 0) & (local_cols < logits.shape[-1])
    if not bool(in_shard.any()):
        return
    sel_rows = rows.to(torch.int64)[in_shard]
    sel_cols = local_cols[in_shard]
    logits[sel_rows, sel_cols] = -float("inf")


def test_shard_split_equals_full():
    print("[1] sharded MinTokens mask == full-vocab mask...")
    torch.manual_seed(0)
    B, V, TP = 6, 4096, 8
    assert V % TP == 0
    width = V // TP

    base = torch.randn(B, V)

    # Stock: censor a set of (row, eos/stop id) over the full tensor.
    # Rows 0,2,5 are under-min; each has stop tokens {199, 200} plus EOS 201087%V.
    rows = torch.tensor([0, 0, 2, 2, 5, 5], dtype=torch.int32)
    toks = torch.tensor([199, 4095, 200, 12, 0, 4094], dtype=torch.int32)

    full = base.clone()
    full[rows.long(), toks.long()] = -float("inf")

    # Sharded: reconstruct by masking each shard independently, then concat.
    shards = []
    for r in range(TP):
        vs = r * width
        shard = base[:, vs:vs + width].clone()
        _apply_min_tokens_mask(shard, vs, (rows, toks))
        shards.append(shard)
    recon = torch.cat(shards, dim=-1)

    # -inf positions must match exactly, finite values untouched.
    full_inf = torch.isinf(full) & (full < 0)
    recon_inf = torch.isinf(recon) & (recon < 0)
    if not bool((full_inf == recon_inf).all()):
        print("  FAIL: -inf mask positions differ between full and sharded")
        return False
    # Every censored (row,tok) accounted for exactly once.
    if int(recon_inf.sum()) != rows.numel():
        print(f"  FAIL: expected {rows.numel()} censored, got {int(recon_inf.sum())}")
        return False
    # Finite entries identical.
    fin = ~full_inf
    if not torch.allclose(full[fin], recon[fin]):
        print("  FAIL: finite logits changed by sharded masking")
        return False
    print(f"  PASS ({rows.numel()} stop tokens censored, split across {TP} shards)")
    return True


def test_empty_and_none():
    print("[2] empty slice / None are no-ops...")
    x = torch.randn(4, 512)
    ref = x.clone()
    _apply_min_tokens_mask(x, 0, None)
    _apply_min_tokens_mask(x, 0, (torch.tensor([], dtype=torch.int32),
                                  torch.tensor([], dtype=torch.int32)))
    if not torch.equal(x, ref):
        print("  FAIL: no-op path modified logits")
        return False
    print("  PASS")
    return True


def test_token_outside_shard_ignored():
    print("[3] token id outside this shard is not masked here...")
    x = torch.randn(2, 100)
    ref = x.clone()
    # vocab_start=500, width=100 -> shard owns [500,600). Token 50 (rank 0) absent.
    _apply_min_tokens_mask(x, 500, (torch.tensor([0]), torch.tensor([50])))
    if not torch.equal(x, ref):
        print("  FAIL: masked a token not owned by this shard")
        return False
    # Token 550 IS owned -> local col 50.
    _apply_min_tokens_mask(x, 500, (torch.tensor([1]), torch.tensor([550])))
    if not (torch.isinf(x[1, 50]) and x[1, 50] < 0):
        print("  FAIL: did not mask owned token 550 -> local 50")
        return False
    print("  PASS")
    return True


def main():
    print("=" * 70)
    print("Dist-Sample MinTokens shard-masking correctness")
    print("=" * 70)
    tests = [test_shard_split_equals_full, test_empty_and_none,
             test_token_outside_shard_ignored]
    results = []
    for t in tests:
        try:
            results.append(t())
        except Exception:
            import traceback
            traceback.print_exc()
            results.append(False)
        print()
    failed = sum(1 for r in results if not r)
    print("=" * 70)
    print(f"{len(results) - failed} passed, {failed} failed")
    if failed:
        print("✗ FAILED")
        return 1
    print("✓ PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
