#!/bin/bash
# Phase C verification suite for the single-spin (Option 2) projected solver
# with b_comm_size > 1 (branch gdb-config-structure).
#
# The property under test: CONVERGED ENERGIES MUST NOT DEPEND ON RANK COUNT.
# Everything else here supports that claim or guards a failure mode that is
# silent (wrong number, no error) rather than loud.
#
# Usage:
#   ./run_phase_c.sh <fcidump> <detfile-dir> [diag-binary]
#
# detfile-dir must contain det_top50.txt, det_top100.txt, det_top1000.txt as
# produced by single_spin_workflow/make_n2_dets.py. Exit status 0 = all passed.
#
# NOTE on tolerance: comparisons are to 1e-8 Ha, not bitwise. The solver's own
# tolerance is 1e-4, and codegen differences move the last digits by ~1e-13 (see
# the Phase B2 commit message). Bitwise equality is the wrong test here.

set -u
FCI=${1:?usage: run_phase_c.sh <fcidump> <detfile-dir> [diag]}
DDIR=${2:?usage: run_phase_c.sh <fcidump> <detfile-dir> [diag]}
DIAG=${3:-$(dirname "$0")/../../apps/chemistry_gdb_selected_basis_diagonalization/diag}

[[ -x $DIAG ]] || { echo "FATAL: diag binary not executable: $DIAG"; exit 2; }
[[ -r $FCI  ]] || { echo "FATAL: cannot read fcidump: $FCI"; exit 2; }

# Sections 7 and 8 must run each rank in its own directory (the RDM and
# wavefunction writers emit fixed filenames into the cwd), so every path used
# after a `cd` has to be absolute. Resolving them here rather than at the point
# of use: a relative $DIAG silently became "command not found" inside those
# subshells, which the harness reported as "no energy produced" -- a test-harness
# bug that looked exactly like a solver failure.
_abspath() { ( cd "$(dirname "$1")" && printf '%s/%s\n' "$(pwd -P)" "$(basename "$1")" ); }
DIAG=$(_abspath "$DIAG")
FCI=$(_abspath "$FCI")
DDIR=$( cd "$DDIR" && pwd -P )

: "${OMP_NUM_THREADS:=2}"
export OMP_NUM_THREADS
TOL=1e-8
PASS=0; FAIL=0; SKIP=0
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

# --- helpers ---------------------------------------------------------------
# energy of the last "sbd: Energy =" line, or empty on failure
_energy() { grep -o 'sbd: Energy = *[-0-9.]*' "$1" | tail -1 | grep -o '[-0-9.]*$'; }
_csfdim() { grep -o 'projected CSF dim = [0-9]*' "$1" | tail -1 | grep -o '[0-9]*$'; }

_close() {  # _close a b tol -> 0 if |a-b| <= tol
  awk -v a="$1" -v b="$2" -v t="$3" \
      'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<=t)}'
}

# run <np> <b_comm> <detfile> <extra args...>  -> writes $TMP/out, echoes energy
_run() {
  local np=$1 bc=$2 det=$3; shift 3
  mpirun -np "$np" "$DIAG" --fcidump "$FCI" --detfiles "$det" \
         --b_comm_size "$bc" "$@" > "$TMP/out" 2>&1
  _energy "$TMP/out"
}

_check() {  # _check <label> <got> <want>
  local label=$1 got=$2 want=$3
  if [[ -z $got ]]; then
    printf '  FAIL  %-52s no energy produced\n' "$label"
    grep -iE 'error|abort|assert' "$TMP/out" | head -2 | sed 's/^/          /'
    FAIL=$((FAIL+1)); return 1
  fi
  if _close "$got" "$want" "$TOL"; then
    printf '  ok    %-52s %s\n' "$label" "$got"
    PASS=$((PASS+1)); return 0
  else
    printf '  FAIL  %-52s got %s want %s\n' "$label" "$got" "$want"
    FAIL=$((FAIL+1)); return 1
  fi
}

_check_int() {  # _check_int <label> <got> <want>
  if [[ "$2" == "$3" ]]; then
    printf '  ok    %-52s %s\n' "$1" "$2"; PASS=$((PASS+1))
  else
    printf '  FAIL  %-52s got %s want %s\n' "$1" "$2" "$3"; FAIL=$((FAIL+1))
  fi
}

# Oracle values (see oracle_refs.txt in this directory).
S50=-108.667515671
S100=-108.729420078
S1000=-108.792795252
T50=-108.639031853

echo "=========================================================================="
echo " Phase C: single-spin projected solver, b_comm sweep"
echo " diag = $DIAG"
echo " OMP_NUM_THREADS=$OMP_NUM_THREADS   tolerance=$TOL Ha"
echo "=========================================================================="

# --- 1. rank-count independence, both matvec methods -----------------------
echo
echo "[1] Energies independent of b_comm, vs the DENSE ORACLE (not vs SBD itself)"
for m in 0 1; do
  for bc in 1 2 4; do
    e=$(_run "$bc" "$bc" "$DDIR/det_top50.txt"  --method $m --single_spin 1 \
             --do_redist_config 1 --iteration 300)
    _check "top50  method=$m b_comm=$bc" "$e" "$S50"
    c=$(_csfdim "$TMP/out"); _check_int "  ^ global CSF dim" "$c" "56"

    e=$(_run "$bc" "$bc" "$DDIR/det_top100.txt" --method $m --single_spin 1 \
             --do_redist_config 1 --iteration 300)
    _check "top100 method=$m b_comm=$bc" "$e" "$S100"
    c=$(_csfdim "$TMP/out"); _check_int "  ^ global CSF dim" "$c" "117"
  done
done
for bc in 1 2 4; do
  e=$(_run "$bc" "$bc" "$DDIR/det_top1000.txt" --method 0 --single_spin 1 \
           --do_redist_config 1 --iteration 400)
  _check "top1000 method=0 b_comm=$bc" "$e" "$S1000"
  c=$(_csfdim "$TMP/out"); _check_int "  ^ global CSF dim" "$c" "3906"
done

# --- 2. a spin sector other than the ground state --------------------------
# top50's true ground state is an S=2 quintet at -108.71257; a solver that
# quietly ignored the projection would return that, not the triplet.
echo
echo "[2] Triplet sector (ground state here is a QUINTET -- catches no-op projection)"
for bc in 1 2; do
  e=$(_run "$bc" "$bc" "$DDIR/det_top50.txt" --method 0 --single_spin 3 \
           --do_redist_config 1 --iteration 300)
  _check "top50 triplet b_comm=$bc" "$e" "$T50"
  c=$(_csfdim "$TMP/out"); _check_int "  ^ global CSF dim" "$c" "80"
done

# --- 3. the parallel / fused-MGS path -------------------------------------
# K is the LOCAL slice length, so distributing SHRINKS it and these branches
# become unreachable in exactly the multi-rank runs that need testing. The
# override forces them. Verified separately that deleting the fused-MGS
# allreduce breaks this case (-333.18) and leaves the default path correct,
# i.e. the override really does reach the code it claims to.
echo
echo "[3] Forced parallel + fused-MGS path (SBD_SS_PAR_THRESHOLD=0)"
export SBD_SS_PAR_THRESHOLD=0
for bc in 1 2 4; do
  e=$(_run "$bc" "$bc" "$DDIR/det_top100.txt" --method 0 --single_spin 1 \
           --do_redist_config 1 --iteration 300)
  _check "top100 thr=0 b_comm=$bc" "$e" "$S100"
done
for bc in 1 2; do
  e=$(_run "$bc" "$bc" "$DDIR/det_top1000.txt" --method 0 --single_spin 1 \
           --do_redist_config 1 --iteration 400)
  _check "top1000 thr=0 b_comm=$bc" "$e" "$S1000"
done
unset SBD_SS_PAR_THRESHOLD

# --- 4. thread count, with MPI-inside-OpenMP active ------------------------
echo
echo "[4] Thread sweep at b_comm=2 with the fused MGS forced (omp master + MPI)"
export SBD_SS_PAR_THRESHOLD=0
_saved_threads=$OMP_NUM_THREADS
for th in 1 2 4; do
  export OMP_NUM_THREADS=$th
  e=$(_run 2 2 "$DDIR/det_top1000.txt" --method 0 --single_spin 1 \
           --do_redist_config 1 --iteration 400)
  _check "top1000 b_comm=2 threads=$th" "$e" "$S1000"
done
export OMP_NUM_THREADS=$_saved_threads
unset SBD_SS_PAR_THRESHOLD

# --- 5. multi-root: all roots must be singlets AND match the oracle -------
echo
echo "[5] Multi-root, b_comm 1 vs 2, each root vs the exact singlet ladder"
declare -a WANT=(-108.729420078 -108.571828872 -108.540036804 -108.512009005)
for bc in 1 2; do
  mpirun -np $bc "$DIAG" --fcidump "$FCI" --detfiles "$DDIR/det_top100.txt" \
    --b_comm_size $bc --do_redist_config 1 --method 0 --single_spin 1 \
    --nroots 4 --block 24 --iteration 600 > "$TMP/mr" 2>&1
  for p in 0 1 2 3; do
    g=$(grep -o "SingleSpin E\[$p\] = [-0-9.]*" "$TMP/mr" | tail -1 | grep -o '[-0-9.]*$')
    _check "top100 4-root b_comm=$bc root[$p]" "$g" "${WANT[$p]}"
  done
done

# --- 6. thick restart: the Hv-carry invariant ----------------------------
# The Hv carry bug was invisible at small K and only appeared at K=113394, so
# this is a necessary-not-sufficient check. Run it at the largest K available.
echo
echo "[6] --restart_keep sweep at b_comm=2 (Hv carry must stay = H v)"
for rk in 1 3 7 15 25; do
  e=$(_run 2 2 "$DDIR/det_top1000.txt" --method 0 --single_spin 1 \
           --do_redist_config 1 --iteration 600 --restart_keep $rk)
  _check "top1000 b_comm=2 restart_keep=$rk" "$e" "$S1000"
done

# --- 7. RDMs and the 1-RDM trace -----------------------------------------
echo
echo "[7] RDMs agree across rank count; trace(1pRDM) = nelec"
for bc in 1 2; do
  W="$TMP/rdm$bc"; mkdir -p "$W"
  ( cd "$W" && mpirun -np $bc "$DIAG" --fcidump "$FCI" \
      --detfiles "$DDIR/det_top100.txt" --b_comm_size $bc --do_redist_config 1 \
      --method 0 --single_spin 1 --iteration 300 --rdm 1 > log 2>&1 )
done
if [[ -r $TMP/rdm1/1pRDM.txt && -r $TMP/rdm2/1pRDM.txt ]]; then
  tr1=$(awk '$1==$2{s+=$(NF)} END{printf "%.9f", s}' "$TMP/rdm1/1pRDM.txt")
  tr2=$(awk '$1==$2{s+=$(NF)} END{printf "%.9f", s}' "$TMP/rdm2/1pRDM.txt")
  _check "trace(1pRDM) b_comm=1" "$tr1" "10.0"
  _check "trace(1pRDM) b_comm=2" "$tr2" "10.0"
  # 1pRDM is "i j value" (3 cols) but 2pRDM is "i j k l value" (5 cols), so
  # compare the LAST field of each half rather than a fixed column index --
  # hardcoding column 3 silently compared an INDEX against a value and reported
  # a bogus max|diff| of 15.
  for f in 1pRDM 2pRDM; do
    d=$(paste "$TMP/rdm1/$f.txt" "$TMP/rdm2/$f.txt" \
        | awk '{na=NF/2; v=$(na)-$(NF); if(v<0)v=-v; if(v>m)m=v} END{printf "%.3e", m+0}')
    if _close "$d" 0 1e-9; then
      printf '  ok    %-52s max|diff|=%s\n' "$f b_comm=1 vs 2" "$d"; PASS=$((PASS+1))
    else
      printf '  FAIL  %-52s max|diff|=%s\n' "$f b_comm=1 vs 2" "$d"; FAIL=$((FAIL+1))
    fi
  done
else
  echo "  SKIP  RDM files not produced"; SKIP=$((SKIP+1))
fi

# --- 8. wavefunction save/load round-trip --------------------------------
echo
echo "[8] Wavefunction save/load round-trip (nothing else covers this)"
for bc in 1 2; do
  W="$TMP/wf$bc"; mkdir -p "$W"
  ( cd "$W" && mpirun -np $bc "$DIAG" --fcidump "$FCI" \
      --detfiles "$DDIR/det_top100.txt" --b_comm_size $bc --do_redist_config 1 \
      --method 0 --single_spin 1 --iteration 300 --savename wf > save 2>&1
    mpirun -np $bc "$DIAG" --fcidump "$FCI" \
      --detfiles "$DDIR/det_top100.txt" --b_comm_size $bc --do_redist_config 1 \
      --method 0 --single_spin 1 --iteration 300 --loadname wf_root0 --init 2 \
      > load 2>&1 )
  _check "round-trip b_comm=$bc (reload)" "$(_energy "$W/load")" "$S100"
done

# --- 9. the guards must still fire --------------------------------------
echo
echo "[9] Negative tests: bad input must ABORT, not return a plausible number"
# 9a. b_comm>1 without config-aligned redistribution -> split blocks
mpirun -np 2 "$DIAG" --fcidump "$FCI" --detfiles "$DDIR/det_top100.txt" \
  --b_comm_size 2 --method 0 --single_spin 1 --iteration 50 > "$TMP/g1" 2>&1
if grep -q "INCOMPLETE configuration block" "$TMP/g1"; then
  printf '  ok    %-52s aborted as expected\n' "b_comm=2 without --do_redist_config"
  PASS=$((PASS+1))
else
  printf '  FAIL  %-52s did NOT abort (energy=%s)\n' \
    "b_comm=2 without --do_redist_config" "$(_energy "$TMP/g1")"
  FAIL=$((FAIL+1))
fi
# 9b. spin-incomplete determinant list
if [[ -r $DDIR/det_incomplete.txt ]]; then
  mpirun -np 1 "$DIAG" --fcidump "$FCI" --detfiles "$DDIR/det_incomplete.txt" \
    --method 0 --single_spin 1 --iteration 50 > "$TMP/g2" 2>&1
  if grep -q "INCOMPLETE configuration block" "$TMP/g2"; then
    printf '  ok    %-52s aborted as expected\n' "spin-incomplete det list"
    PASS=$((PASS+1))
  else
    printf '  FAIL  %-52s did NOT abort\n' "spin-incomplete det list"; FAIL=$((FAIL+1))
  fi
else
  echo "  SKIP  det_incomplete.txt not present (see make_negative_dets.py)"; SKIP=$((SKIP+1))
fi
# 9c. mixed-Sz determinant list
if [[ -r $DDIR/det_mixedsz.txt ]]; then
  mpirun -np 1 "$DIAG" --fcidump "$FCI" --detfiles "$DDIR/det_mixedsz.txt" \
    --method 0 --single_spin 1 --iteration 50 > "$TMP/g3" 2>&1
  if grep -q "mixes spin projections" "$TMP/g3"; then
    printf '  ok    %-52s aborted as expected\n' "mixed-Sz det list"
    PASS=$((PASS+1))
  else
    printf '  FAIL  %-52s did NOT abort\n' "mixed-Sz det list"; FAIL=$((FAIL+1))
  fi
else
  echo "  SKIP  det_mixedsz.txt not present (see make_negative_dets.py)"; SKIP=$((SKIP+1))
fi

echo
echo "=========================================================================="
printf " passed %d   failed %d   skipped %d\n" "$PASS" "$FAIL" "$SKIP"
echo "=========================================================================="
[[ $FAIL -eq 0 ]]
