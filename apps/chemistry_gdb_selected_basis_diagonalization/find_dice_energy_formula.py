#!/usr/bin/env python3
"""
Find the correct energy formula for DICE spatial RDMs.
Target: -108.843540
"""

from compute_energy_from_rdm import parse_fcidump, read_1prdm, read_2prdm, get_two_body_integral

fcidump_file = 'N2/FCIDUMP_N2_6-31G_R2.0_10e16o.dat'
norb, nelec, core_energy, one_body_integrals, two_body_integrals = parse_fcidump(fcidump_file)

rdm1_dice = read_1prdm('N2/spatial1pRDM.txt', norb)
rdm2_dice = read_2prdm('N2/spatial2pRDM.txt', norb)

target = -108.843540

print(f"Target energy: {target}")
print(f"Core energy: {core_energy}")
print("="*70)

# One-body is straightforward
one_body = sum(one_body_integrals.get((i,j), one_body_integrals.get((j,i), 0)) * rdm1_dice[i,j] 
               for i in range(norb) for j in range(norb))

print(f"One-body energy: {one_body:.10f}")
print()

# Try different two-body contractions
contractions = [
    ("v[io,ia,jo,ja] * rdm2[io,jo,ia,ja]", 
     lambda io,jo,ia,ja: get_two_body_integral(two_body_integrals, io, ia, jo, ja) * rdm2_dice[io,jo,ia,ja]),
    
    ("v[io,jo,ia,ja] * rdm2[io,jo,ia,ja]", 
     lambda io,jo,ia,ja: get_two_body_integral(two_body_integrals, io, jo, ia, ja) * rdm2_dice[io,jo,ia,ja]),
    
    ("v[io,ia,jo,ja] * rdm2[io,ia,jo,ja]", 
     lambda io,jo,ia,ja: get_two_body_integral(two_body_integrals, io, ia, jo, ja) * rdm2_dice[io,ia,jo,ja]),
    
    ("v[io,jo,ia,ja] * rdm2[io,ia,jo,ja]", 
     lambda io,jo,ia,ja: get_two_body_integral(two_body_integrals, io, jo, ia, ja) * rdm2_dice[io,ia,jo,ja]),
]

print("Testing different contractions with factor 0.5:")
print("-"*70)

best_diff = float('inf')
best_result = None

for desc, contraction_func in contractions:
    two_body = 0.0
    for io in range(norb):
        for jo in range(norb):
            for ia in range(norb):
                for ja in range(norb):
                    two_body += 0.5 * contraction_func(io, jo, ia, ja)
    
    total = core_energy + one_body + two_body
    diff = abs(total - target)
    
    print(f"{desc:50s}")
    print(f"  Two-body: {two_body:12.6f}, Total: {total:12.6f}, Diff: {diff:.6f}")
    
    if diff < best_diff:
        best_diff = diff
        best_result = (desc, two_body, total)

print("\n" + "="*70)
if best_result and best_diff < 0.01:
    print(f"✓ MATCH FOUND!")
    print(f"  Formula: {best_result[0]}")
    print(f"  Total energy: {best_result[2]:.10f}")
    print(f"  Difference from target: {best_diff:.2e}")
else:
    print(f"No exact match found. Best result:")
    if best_result:
        print(f"  Formula: {best_result[0]}")
        print(f"  Total energy: {best_result[2]:.10f}")
        print(f"  Difference: {best_diff:.6f}")

# Made with Bob
