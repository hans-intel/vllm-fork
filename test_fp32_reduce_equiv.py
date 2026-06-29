#!/usr/bin/env python3
"""Verify the fp32 0-fill SUM reduce picks the SAME cross-rank winner as int64 MAX.

dist-sample's cross-rank reduce must pick, per token, the (value, index) with the
max gumbel-perturbed value across TP shards. Two implementations:
  - int64 MAX: pack (31-bit value key)<<31 | idx, all_reduce(MAX), unpack idx.
  - fp32 SUM : 0-fill [B,TP,2] (value,idx), all_reduce(SUM), local argmax over TP.
This test simulates TP ranks on host (no XPU/collective needed): builds random
per-rank winners, computes both reducers, asserts identical chosen index. Covers
the failure modes: negative values (0-fill must not beat a real negative winner),
ties, large indices near 2**24 fp32 boundary.
"""
import sys
import torch


def _pack(value, idx):
    """int64 packing matching _dist_gumbel_packed_kernel / get_top_tokens."""
    val_i32 = value.to(torch.float32).view(torch.int32)
    key_i32 = torch.where(val_i32 >= 0, val_i32 | (-2147483648), ~val_i32)
    key_u32 = key_i32.to(torch.int64) & 0xFFFFFFFF
    key31 = key_u32 >> 1
    return (key31 << 31) | (idx.to(torch.int64) & 0x7FFFFFFF)


def _int64_winner(values, idxs):
    """values,idxs: [B,TP]. Returns chosen idx [B] via packed int64 MAX."""
    packed = _pack(values, idxs)            # [B,TP]
    winner = packed.max(dim=-1).values      # cross-rank MAX
    return (winner & 0x7FFFFFFF)


def _fp32_winner(values, idxs):
    """0-fill SUM then local argmax. table[:,r,:] holds rank r's (val,idx)."""
    B, TP = values.shape
    table = torch.zeros(B, TP, 2, dtype=torch.float32)
    for r in range(TP):
        table[:, r, 0] = values[:, r]
        table[:, r, 1] = idxs[:, r].to(torch.float32)
    # SUM all-reduce is identity here (each slot filled once); argmax over TP.
    best = table[:, :, 0].argmax(dim=-1, keepdim=True)
    return table[:, :, 1].gather(-1, best).squeeze(-1).to(torch.int64)


def test_random(name, values, idxs):
    a = _int64_winner(values, idxs)
    b = _fp32_winner(values, idxs)
    mism = int((a != b).sum())
    if mism:
        bad = (a != b).nonzero().flatten()[:5].tolist()
        print(f"  FAIL [{name}]: {mism} mismatches, e.g. rows {bad}")
        for r in bad:
            print(f"    row {r}: vals={values[r].tolist()} idxs={idxs[r].tolist()} "
                  f"int64->{a[r].item()} fp32->{b[r].item()}")
        return False
    print(f"  PASS [{name}] ({values.shape[0]} tokens, TP={values.shape[1]})")
    return True


def main():
    print("=" * 70)
    print("fp32 SUM-0fill reduce == int64 MAX reduce (winner equivalence)")
    print("=" * 70)
    torch.manual_seed(0)
    B, TP, V = 4096, 8, 201088
    ok = True

    # 1. realistic gumbel-perturbed values (logit ~N(0,5) + Gumbel), large idxs.
    vals = torch.randn(B, TP) * 6.0
    idxs = torch.randint(0, V, (B, TP), dtype=torch.int64)
    ok &= test_random("random positive+negative", vals, idxs)

    # 2. all-negative values (0-fill MUST NOT win — the bug a naive SUM could hit).
    vals_neg = -torch.rand(B, TP) * 10.0 - 0.1
    ok &= test_random("all-negative values", vals_neg, idxs)

    # 3. indices right at the fp32-exact boundary (2**24 = 16777216 > V, so safe;
    #    test the largest real vocab ids).
    idx_hi = torch.randint(V - 100, V, (B, TP), dtype=torch.int64)
    ok &= test_random("high indices near vocab max", vals, idx_hi)

    # 4. exact ties in value across ranks (int64 breaks ties by larger idx via the
    #    low bits; fp32 argmax takes the FIRST max. These CAN differ — verify the
    #    chosen token is still a legitimate max-value winner, not garbage).
    vals_tie = torch.zeros(B, TP)
    a = _int64_winner(vals_tie, idxs)
    b = _fp32_winner(vals_tie, idxs)
    # both must return an index that WAS one of the tied candidates for that row.
    a_valid = torch.tensor([a[i].item() in idxs[i].tolist() for i in range(B)]).all()
    b_valid = torch.tensor([b[i].item() in idxs[i].tolist() for i in range(B)]).all()
    if a_valid and b_valid:
        agree = int((a == b).sum())
        print(f"  PASS [exact ties] both pick a valid tied index "
              f"({agree}/{B} happen to agree; tie-break differs harmlessly)")
    else:
        print(f"  FAIL [exact ties]: invalid index returned")
        ok = False

    print("=" * 70)
    if ok:
        print("✓ fp32 reduce picks the same winner as int64 MAX (ties harmless)")
        return 0
    print("✗ FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
