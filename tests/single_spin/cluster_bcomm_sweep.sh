#!/bin/bash
# Cluster sweep: find the wall-clock optimum over --b_comm_size at fixed np.
#
# Why sweep at all: Phase B2 distributes the CSF/Krylov space over b_comm ONLY.
# h_comm_size = np/(b_comm*t_comm), and the determinant list is broadcast across
# h_comm and t_comm (main.cc:167-169), so --b_comm_size 1 replicates everything
# and gets zero benefit from B2. But b_comm also carries the matvec's ket
# rotation, so bigger is not automatically faster -- hence a measurement, not a
# guess.
#
# Usage: ./cluster_bcomm_sweep.sh <fcidump> <detfile> [np] [omp] [nroots]
set -u
FCI=${1:?usage: cluster_bcomm_sweep.sh <fcidump> <detfile> [np] [omp] [nroots]}
DET=${2:?}
NP=${3:-48}
OMP=${4:-4}
NROOTS=${5:-1}
DIAG=${DIAG:-$(dirname "$0")/../../apps/chemistry_gdb_selected_basis_diagonalization/diag}

export OMP_NUM_THREADS=$OMP
export SBD_SS_TIMING=1     # per-phase breakdown; "end davidson" is NOT solve time

echo "np=$NP omp=$OMP nroots=$NROOTS  det=$(basename "$DET")"
printf '%-8s %-8s %-14s %-14s %-14s %s\n' \
       b_comm h_comm energy solve_s total_s note

for BC in 1 2 4 6 12 24 48; do
  (( NP % BC == 0 )) || continue
  H=$(( NP / BC ))
  EXTRA=""
  (( BC > 1 )) && EXTRA="--do_redist_config 1"
  LOG=sweep_b${BC}.log
  /usr/bin/time -p mpirun -np "$NP" "$DIAG" --fcidump "$FCI" --detfiles "$DET" \
      --b_comm_size "$BC" --method 0 --single_spin 1 --nroots "$NROOTS" \
      --block 24 --iteration 600 $EXTRA > "$LOG" 2>&1
  E=$(grep -o 'sbd: Energy = *[-0-9.]*' "$LOG" | tail -1 | grep -o '[-0-9.]*$')
  # "end davidson" brackets the projector build and per-root postprocessing too,
  # so report the loop total from SBD_SS_TIMING when present.
  SOLVE=$(grep -o 'LOOP TOTAL *= *[0-9.]*' "$LOG" | tail -1 | grep -o '[0-9.]*$')
  TOT=$(grep -E '^real' "$LOG" | tail -1 | awk '{print $2}')
  NOTE=""
  [[ -z $E ]] && NOTE=$(grep -iEm1 'error|abort' "$LOG" | cut -c1-60)
  printf '%-8s %-8s %-14s %-14s %-14s %s\n' \
         "$BC" "$H" "${E:-FAIL}" "${SOLVE:--}" "${TOT:--}" "$NOTE"
done

echo
echo "All energies in the table must agree to ~1e-8 Ha. If one differs, that"
echo "configuration is WRONG, not merely slower -- stop and report it."
