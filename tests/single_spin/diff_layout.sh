#!/bin/zsh
# Differential layout comparison: run the same problem under a REFERENCE MPI
# decomposition (known correct) and a SUSPECT one, and report the first inner step
# where the physically-comparable quantities diverge.
#
# Chunk 2 of the hunt for the b_comm>1 + h_comm>1 + OMP>=3 wrong-energy bug in
# --single_spin --method 1. Deliberately reports the first diverging OPERATION and
# nothing more: ten mechanism hypotheses have been falsified by now, so the method
# here is measurement, not inference.
#
# Both layouts use the same np, so h_comm_size = np/(b_comm*t_comm) is the only
# thing that changes:
#   reference   b_comm=np           -> h_comm=1   (verified correct at any OMP)
#   suspect     b_comm=NB           -> h_comm=np/NB
# The reference is run ONCE at OMP=1 -- it is deterministic and layout-invariant
# in E/Ritz to 1e-15 (see fp_compare.py for what that means and why v/Hv are not
# compared). The suspect is run REPEATS times, because the failure rate is roughly
# 3-in-4 and a single lucky run has misled this investigation twice already.
#
# MEASURED (top8000, np=4, ref b=4/h=1 OMP=1, suspect b=2/h=2 OMP=3, method 1):
#   4/4 runs diverge. The error sits at the 1e-13 noise floor for the first 10-17
#   inner steps, then jumps by x119..x795 in a SINGLE step and ramps geometrically
#   to 1e-1 over the next four. The origin step is not fixed across repeats --
#   1.7, 0.11, 0.10, 0.11 -- but every one has ib >= 10, i.e. a large Krylov basis
#   late in an inner cycle. Two of them are in outer iteration 0, so thick restart
#   is not involved (consistent with --iteration 1 also failing).
#
# usage: ./diff_layout.sh [DETFILE] [NP] [NB] [OMP] [REPEATS]
# env:   FCI=<fcidump>  ITER=<n>  EXTRA="<extra diag flags>"
#        CURVE=--curve   print the per-step error growth curve for every run.
#                        Read this before concluding a step is at fault: see the
#                        header of fp_compare.py on crossings vs origins.

set -u
HERE=${0:A:h}
DIAG=${DIAG:-${HERE}/../../apps/chemistry_gdb_selected_basis_diagonalization/diag}
DIAG=${DIAG:A}                      # absolute: sections below may cd
FCI=${FCI:-$HOME/Downloads/for_claude_n2/N2_R2.0_6-31g_10e_16o.dat}

DET=${1:-/tmp/n2reg/det_top8000.txt}
NP=${2:-4}
NB=${3:-2}
OMP=${4:-3}
REPEATS=${5:-5}
ITER=${ITER:-300}
EXTRA=${EXTRA:-}

[[ -x $DIAG ]] || { print -u2 "no diag binary at $DIAG"; exit 2 }
[[ -r $FCI  ]] || { print -u2 "no fcidump at $FCI";      exit 2 }
[[ -r $DET  ]] || { print -u2 "no detfile at $DET";      exit 2 }

WORK=$(mktemp -d /tmp/difflayout.XXXXXX)
trap 'rm -rf $WORK' EXIT
rm -f /tmp/fp_bad.log /tmp/fp_ref.log   # never compare against a previous run's

# SBD_SS_PAR_THRESHOLD=0 forces the CSF-space parallel and fused-MGS paths on.
# Without it K is the LOCAL slice length, so distributing can drop every rank
# below the 4096 gate and leave exactly the code under suspicion unexecuted.
common=(--fcidump $FCI --detfiles $DET --do_redist_config 1
        --method 1 --single_spin 1 --iteration $ITER ${=EXTRA})

run() {  # run OUTFILE NPROC BCOMM OMPTHREADS
  local out=$1 np=$2 bc=$3 omp=$4
  SBD_SS_FP=1 OMP_NUM_THREADS=$omp SBD_SS_PAR_THRESHOLD=0 \
    mpirun -np $np $DIAG $common --b_comm_size $bc --t_comm_size 1 \
    > $out 2>&1
  print $?
}

energy() { grep -o 'sbd: Energy = *[-0-9.]*' $1 | tail -1 | grep -o '[-0-9.]*$' }

print "detfile   $DET ($(wc -l < $DET | tr -d ' ') dets)"
print "reference np=$NP b_comm=$NP  -> h_comm=1        OMP=1"
print "suspect   np=$NP b_comm=$NB  -> h_comm=$((NP/NB))        OMP=$OMP   x$REPEATS"
print ""

print -n "reference ... "
rrc=$(run $WORK/ref.log $NP $NP 1)
rE=$(energy $WORK/ref.log)
print "rc=$rrc E=${rE:-NONE} steps=$(grep -c '\[fp\] H ' $WORK/ref.log)"
if [[ -z $rE ]]; then
  print -u2 "reference produced no energy -- cannot diff against it"
  grep -iE 'error|abort' $WORK/ref.log | head -5
  exit 2
fi
print ""

fails=0
for i in {1..$REPEATS}; do
  print -n "run $i/$REPEATS ... "
  src=$(run $WORK/t$i.log $NP $NB $OMP)
  tE=$(energy $WORK/t$i.log)
  printf "rc=%s E=%-20s " "$src" "${tE:-NONE}"
  python3 $HERE/fp_compare.py ${=CURVE:-} $WORK/ref.log $WORK/t$i.log
  (( $? != 0 )) && (( fails++ ))
  # Keep the log of the FIRST failing run; it is the input to the next chunk.
  # Guard on "not already saved" -- an unguarded copy here overwrites the kept
  # log on every later iteration, so a passing final run silently replaces the
  # failing one and the saved pair then compares clean. That happened.
  if [[ ! -e /tmp/fp_bad.log ]] && { [[ -z $tE ]] || [[ $tE != $rE ]] }; then
    cp $WORK/t$i.log  /tmp/fp_bad.log
    cp $WORK/ref.log  /tmp/fp_ref.log
    print -n "  [saved run $i as /tmp/fp_bad.log] "
  fi
done

print ""
print "$fails/$REPEATS runs diverged."
if (( fails > 0 )); then
  print "kept /tmp/fp_ref.log and /tmp/fp_bad.log for the next chunk."
fi
