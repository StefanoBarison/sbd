/**
@file sbd/chemistry/tpb/mult.h
@brief Function to perform Hamiltonian operation for general determinant basis
*/
#ifndef SBD_CHEMISTRY_GDB_MULT_H
#define SBD_CHEMISTRY_GDB_MULT_H

#include <algorithm>
#include <iostream>
#include <unordered_map>

namespace sbd {
  namespace gdb {
    /// SBD_CHECK_MULT=1 enables the pre-reduction consistency checks in mult().
    inline bool _sbd_check_mult() {
      static const bool on = [](){
        const char* e = std::getenv("SBD_CHECK_MULT");
        return e && e[0] == '1';
      }();
      return on;
    }
    /// Abort with a NAMED diagnostic if `n` differs across `comm`. MPI itself
    /// only reports MPI_ERR_TRUNCATE on an opaque communicator id.
    inline void _sbd_check_same_size(size_t n, MPI_Comm comm,
                                     const char* cname, const char* vname) {
      int csize = 1; MPI_Comm_size(comm, &csize);
      if (csize < 2) return;
      long long mn = static_cast<long long>(n), mx = mn;
      MPI_Allreduce(MPI_IN_PLACE, &mn, 1, MPI_LONG_LONG, MPI_MIN, comm);
      MPI_Allreduce(MPI_IN_PLACE, &mx, 1, MPI_LONG_LONG, MPI_MAX, comm);
      if (mn != mx) {
        int cr = 0, wr = 0;
        MPI_Comm_rank(comm, &cr);
        MPI_Comm_rank(MPI_COMM_WORLD, &wr);
        std::cerr << " sbd: ERROR mult: " << vname << ".size() differs across "
                  << cname << " (min " << mn << ", max " << mx << "); this rank ("
                  << cname << " rank " << cr << ", world rank " << wr << ") has "
                  << n << ". A reduction over " << cname
                  << " would abort with MPI_ERR_TRUNCATE." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
  }
}

namespace sbd {

  namespace gdb {

    template <typename ElemT, typename DetsContainer>
    void mult(const std::vector<ElemT> & hii,
	      const std::vector<ElemT> & wk,
	      std::vector<ElemT> & wb,
	      size_t bit_length,
	      size_t norb,
	      const DetsContainer & det,
	      const DetIndexMap & idxmap,
	      const std::vector<ExcitationLookup> & exidx,
	      const ElemT & I0,
	      const oneInt<ElemT> & I1,
	      const twoInt<ElemT> & I2,
	      MPI_Comm h_comm,
	      MPI_Comm b_comm,
	      MPI_Comm t_comm) {
      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);

      std::vector<ElemT> twk;
      std::vector<ElemT> rwk;
      DetIndexMap tidxmap;
      // std::vector<std::vector<size_t>> tdet;

      if( exidx[0].slide != 0 ) {
	sbd::gdb::MpiSlide(idxmap,tidxmap,-exidx[0].slide,b_comm);
	sbd::MpiSlide(wk,twk,-exidx[0].slide,b_comm);
	// sbd::MpiSlide(det,tdet,-exidx[0].slide,b_comm);
      } else {
	DetIndexMapCopy(idxmap,tidxmap);
	twk = wk;
	// tdet = det;
      }

      if( mpi_rank_t == 0 ) {
	// Local diagonal wb[i] += hii[i]*wk[i]. Index `wk`, the LOCAL ket -- never
	// `twk`, which is the ket rotated in from a neighbouring b rank.
	//
	// History, because this line has been wrong twice. It originally indexed
	// twk and was bounded by twk.size(); --do_redist_config assigns whole
	// configuration orbits, which need not divide evenly (measured on N2 at
	// b_comm=12: one rank holds 45058 determinants, the other eleven 45059), so
	// on the short rank it wrote one element PAST THE END of wb and read past
	// hii -- silent heap corruption every matvec. Downstream MpiAllreduce(wb)
	// passes wb.size() as its count, so the short rank disagreed with its
	// h_comm partner and the run aborted with MPI_ERR_TRUNCATE after diverging.
	// Symptom trio: E[0] from the Rayleigh solve, <w|H|w>, and the final
	// expectation value all DIFFERENT (-34.76 / -18.51 / -16.88), which
	// unconverged-but-consistent iteration cannot produce.
	//
	// That was then patched to min(wb, hii, twk), which stopped the overrun but
	// would silently TRUNCATE the diagonal -- dropping real matrix elements for
	// a variationally too-high energy with no error -- had it ever fired. It
	// never did: t_comm rank 0 always receives task_begin == 0, hence slide == 0,
	// hence twk == wk here, for every (b,t) with t <= b. Correct only by
	// accident, and only while that partition holds.
	//
	// So: index `wk` explicitly, and ASSERT the sizes rather than clamping them.
	// A clamp hides a broken invariant; an assert reports it.
	if( wk.size() != wb.size() || hii.size() < wb.size() ) {
	  int wr = 0; MPI_Comm_rank(MPI_COMM_WORLD,&wr);
	  std::cerr << " sbd: ERROR mult diagonal: size mismatch wb=" << wb.size()
		    << " wk=" << wk.size() << " hii=" << hii.size()
		    << " on world rank " << wr << std::endl;
	  MPI_Abort(MPI_COMM_WORLD,1);
	}
#pragma omp parallel for
	for(size_t i=0; i < wb.size(); i++) {
	  wb[i] += hii[i] * wk[i];
	}
      }

      for(size_t task=0; task < exidx.size(); task++) {
#pragma omp parallel
	{
	  // Stride by the ACTUAL team size, read inside the region -- not by the
	  // omp_get_max_threads() captured outside it.
	  //
	  // This is a strided partition: thread t handles ia = t, t+stride,
	  // t+2*stride, ... It covers every ia if and only if stride equals the real
	  // team size. omp_get_max_threads() is the size of the NEXT team, not this
	  // one; if the runtime hands out fewer threads (dynamic adjustment -- on by
	  // default in libgomp, off in libomp -- or an if() clause, or nesting) then
	  // residue classes [team, max_threads) are NEVER VISITED and their
	  // Hamiltonian contributions are silently missing. No crash, just a wrong
	  // energy. Deriving the stride from omp_get_num_threads() makes the
	  // partition correct for whatever team is actually granted.
	  //
	  // (MpiInitHybrid calls omp_set_dynamic(0), which makes the two agree in
	  // practice. This does not rely on that: correctness here should not depend
	  // on a setting made in a different file.)
	  const size_t stride = static_cast<size_t>(omp_get_num_threads());
	  size_t thread_id = omp_get_thread_num();
	  size_t ia_begin = thread_id;
	  size_t ia_end = idxmap.AdetToDetLen.size();
	  for(size_t ia=ia_begin; ia < ia_end; ia+=stride) {
	    for(size_t ib=0; ib < idxmap.AdetToDetLen[ia]; ib++) {
	      size_t iast = ia;
	      size_t ibst = idxmap.AdetToBdetSM[ia][ib];
	      size_t idet = idxmap.AdetToDetSM[ia][ib];
	      if( idet % mpi_size_h != mpi_rank_h ) continue;

	      // single alpha excitations
	      if( exidx[task].SelfFromBdetLen[ibst] != 0 ) {
		size_t jbst = exidx[task].SelfFromBdetSM[ibst][0];
		for(size_t ja=0; ja < exidx[task].SinglesFromAdetLen[ia]; ja++) {
		  size_t jast = exidx[task].SinglesFromAdetSM[ia][ja];
		  auto itA = std::lower_bound(&tidxmap.BdetToAdetSM[jbst][0],
					      &tidxmap.BdetToAdetSM[jbst][0]
					      +tidxmap.BdetToDetLen[jbst],
					      jast);
		  if( itA != (&tidxmap.BdetToAdetSM[jbst][0]+tidxmap.BdetToDetLen[jbst])) {
		    size_t idxa = std::distance(&tidxmap.BdetToAdetSM[jbst][0],itA);
		    if( jast != tidxmap.BdetToAdetSM[jbst][idxa] ) continue;
		    size_t jdet = tidxmap.BdetToDetSM[jbst][idxa];
		    ElemT eij = OneExcite(det[idet],bit_length,
					  exidx[task].SinglesAdetCrAnSM[ia][2*ja+0],
					  exidx[task].SinglesAdetCrAnSM[ia][2*ja+1],
					  I1,I2);
		    wb[idet] += eij * twk[jdet];
		  }
		}

		// double alpha excitations
		for(size_t ja=0; ja < exidx[task].DoublesFromAdetLen[ia]; ja++) {
		  size_t jast = exidx[task].DoublesFromAdetSM[ia][ja];
		  auto itA = std::lower_bound(&tidxmap.BdetToAdetSM[jbst][0],
					      &tidxmap.BdetToAdetSM[jbst][0]
					      +tidxmap.BdetToDetLen[jbst],
					      jast);
		  if( itA != (&tidxmap.BdetToAdetSM[jbst][0]+tidxmap.BdetToDetLen[jbst])) {
		    size_t idxa = std::distance(&tidxmap.BdetToAdetSM[jbst][0],itA);
		    if( jast != tidxmap.BdetToAdetSM[jbst][idxa] ) continue;
		    size_t jdet = tidxmap.BdetToDetSM[jbst][idxa];
		    ElemT eij = TwoExcite(det[idet],bit_length,
					  exidx[task].DoublesAdetCrAnSM[ia][4*ja+0],
					  exidx[task].DoublesAdetCrAnSM[ia][4*ja+1],
					  exidx[task].DoublesAdetCrAnSM[ia][4*ja+2],
					  exidx[task].DoublesAdetCrAnSM[ia][4*ja+3],
					  I1,I2);
		    wb[idet] += eij * twk[jdet];
		  }
		}
	      } // if there is same beta string

	      // alpha-beta two-particle excitations
	      for(size_t ja=0; ja < exidx[task].SinglesFromAdetLen[ia]; ja++) {
		size_t jast = exidx[task].SinglesFromAdetSM[ia][ja];
		size_t start_idx = 0;
		size_t end_idx = tidxmap.AdetToDetLen[jast];
		size_t SinglesFromBLen = exidx[task].SinglesFromBdetLen[ibst];
		for(size_t k=0; k < SinglesFromBLen; k++) {
		  size_t jbst = exidx[task].SinglesFromBdetSM[ibst][k];
		  if( start_idx >= end_idx ) break;
		  auto itB = std::lower_bound(&tidxmap.AdetToBdetSM[jast][0]+start_idx,
					      &tidxmap.AdetToBdetSM[jast][0]+end_idx,
					      jbst);
		  if( itB != (&tidxmap.AdetToBdetSM[jast][0]+end_idx) ) {
		    size_t idxb = std::distance(&tidxmap.AdetToBdetSM[jast][0],itB);
		    if( jbst != tidxmap.AdetToBdetSM[jast][idxb] ) continue;
		    start_idx = idxb;
		    if( idxb < end_idx ) {
		      if( tidxmap.AdetToBdetSM[jast][idxb] == jbst ) {
			size_t jdet = tidxmap.AdetToDetSM[jast][idxb];
			ElemT eij = TwoExcite(det[idet],bit_length,
					      exidx[task].SinglesAdetCrAnSM[ia][2*ja+0],
					      exidx[task].SinglesBdetCrAnSM[ibst][2*k+0],
					      exidx[task].SinglesAdetCrAnSM[ia][2*ja+1],
					      exidx[task].SinglesBdetCrAnSM[ibst][2*k+1],
					      I1,I2);
			wb[idet] += eij * twk[jdet];
		      }
		    }
		  }
		}
	      }

	      if( exidx[task].SelfFromAdetLen[iast] != 0 ) {
		size_t jast = exidx[task].SelfFromAdetSM[iast][0];
		// single beta excitations
		for(size_t jb=0; jb < exidx[task].SinglesFromBdetLen[ibst]; jb++) {
		  size_t jbst = exidx[task].SinglesFromBdetSM[ibst][jb];
		  auto itB = std::lower_bound(&tidxmap.AdetToBdetSM[jast][0],
					      &tidxmap.AdetToBdetSM[jast][0]
					      +tidxmap.AdetToDetLen[jast],
					      jbst);
		  if( itB != (&tidxmap.AdetToBdetSM[jast][0]+tidxmap.AdetToDetLen[jast]) ) {
		    size_t idxb = std::distance(&tidxmap.AdetToBdetSM[jast][0],itB);
		    if( tidxmap.AdetToBdetSM[jast][idxb] != jbst ) continue;
		    size_t jdet = tidxmap.AdetToDetSM[jast][idxb];
		    ElemT eij = OneExcite(det[idet],bit_length,
					  exidx[task].SinglesBdetCrAnSM[ibst][2*jb+0],
					  exidx[task].SinglesBdetCrAnSM[ibst][2*jb+1],
					  I1,I2);
		    wb[idet] += eij * twk[jdet];
		  }
		}

		// double beta excitations
		for(size_t jb=0; jb < exidx[task].DoublesFromBdetLen[ibst]; jb++) {
		  size_t jbst = exidx[task].DoublesFromBdetSM[ibst][jb];
		  auto itB = std::lower_bound(&tidxmap.AdetToBdetSM[jast][0],
					      &tidxmap.AdetToBdetSM[jast][0]
					      +tidxmap.AdetToDetLen[jast],
					      jbst);
		  if( itB != (&tidxmap.AdetToBdetSM[jast][0]+tidxmap.AdetToDetLen[jast]) ) {
		    size_t idxb = std::distance(&tidxmap.AdetToBdetSM[jast][0],itB);
		    if( tidxmap.AdetToBdetSM[jast][idxb] != jbst ) continue;
		    size_t jdet = tidxmap.AdetToDetSM[jast][idxb];
		    ElemT eij = TwoExcite(det[idet],bit_length,
					  exidx[task].DoublesBdetCrAnSM[ibst][4*jb+0],
					  exidx[task].DoublesBdetCrAnSM[ibst][4*jb+1],
					  exidx[task].DoublesBdetCrAnSM[ibst][4*jb+2],
					  exidx[task].DoublesBdetCrAnSM[ibst][4*jb+3],
					  I1,I2);
		    wb[idet] += eij * twk[jdet];
		  }
		}

	      } // if there are same alpha

	    } // corresponding beta string loop for bra-side basis
	  } // alpha-based loop for bra-side basis
	} // end omp threading

	if( task != exidx.size()-1 ) {
	  int slide = exidx[task].slide-exidx[task+1].slide;
	  rwk.resize(twk.size());
	  std::memcpy(rwk.data(),twk.data(),twk.size()*sizeof(ElemT));
	  DetIndexMap ridxmap;
	  DetIndexMapCopy(tidxmap,ridxmap);
	  sbd::MpiSlide(rwk,twk,slide,b_comm);
	  sbd::gdb::MpiSlide(ridxmap,tidxmap,slide,b_comm);
	}

      } // end task for loop

      MpiAllreduce(wb,MPI_SUM,t_comm);
      MpiAllreduce(wb,MPI_SUM,h_comm);
    } // end function for mult

    template <typename ElemT>
    void mult(const std::vector<ElemT> & hii,
	      const std::vector<std::vector<size_t*>> & ih,
	      const std::vector<std::vector<size_t*>> & jh,
	      const std::vector<std::vector<ElemT*>> & hij,
	      const std::vector<std::vector<size_t>> & len,
	      const std::vector<int> & slide,
	      const std::vector<ElemT> & wk,
	      std::vector<ElemT> & wb,
	      MPI_Comm h_comm,
	      MPI_Comm b_comm,
	      MPI_Comm t_comm) {
      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
      int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);

      std::vector<ElemT> twk;
      std::vector<ElemT> rwk;
      if( slide.size() != 0 ) {
	if( slide[0] != 0 ) {
	  sbd::MpiSlide(wk,twk,-slide[0],b_comm);
	} else {
	  twk = wk;
	}
      }

      if( mpi_rank_t == 0 ) {
	// Bound on the SMALLEST of the three vectors, not on twk.size().
	//
	// This is the local diagonal term wb[i] += hii[i]*wk[i], so all three
	// must be indexed with the LOCAL determinant count. But twk is the
	// rotated ket: when exidx[0].slide != 0 it was MpiSlide'd in from a
	// neighbouring b rank, so twk.size() is the NEIGHBOUR's count. Those
	// coincide only when every b rank holds equally many determinants.
	//
	// --do_redist_config assigns whole configuration orbits, which cannot
	// divide exactly: measured on N2 at b_comm=12, one rank holds 45058
	// determinants and the other eleven hold 45059. The old bound therefore
	// wrote one element past the end of wb (and read past hii) on the short
	// rank -- silent heap corruption every matvec. Downstream,
	// MpiAllreduce(wb,...) passes wb.size() as its count, so the short rank
	// disagreed with its h_comm partner and the run aborted with
	// MPI_ERR_TRUNCATE after diverging. Symptom trio: E[0] from the Rayleigh
	// solve, <w|H|w>, and the final expectation value all DIFFERENT
	// (-34.76 / -18.51 / -16.88), which unconverged-but-consistent iteration
	// cannot produce.
	// Index `wk` (local ket), not `twk` (rotated). See the matching comment in
	// the matrix-free overload above: t_comm rank 0 always gets task_begin == 0
	// hence slide == 0, so twk == wk here -- an accidental invariant that the
	// old min(wb,hii,twk) clamp was silently relying on. Assert, do not clamp:
	// a clamp would drop real diagonal terms if the invariant ever broke.
	if( wk.size() != wb.size() || hii.size() < wb.size() ) {
	  int wr = 0; MPI_Comm_rank(MPI_COMM_WORLD,&wr);
	  std::cerr << " sbd: ERROR mult diagonal (stored): size mismatch wb="
		    << wb.size() << " wk=" << wk.size() << " hii=" << hii.size()
		    << " on world rank " << wr << std::endl;
	  MPI_Abort(MPI_COMM_WORLD,1);
	}
#pragma omp parallel for
	for(size_t i=0; i < wb.size(); i++) {
	  wb[i] += hii[i] * wk[i];
	}
      }

      for(size_t task=0; task < slide.size(); task++) {
	// The stored Hamiltonian was partitioned into len[task].size() slices by
	// whatever team size was active in qcham (qcham.h:76-80). This region must
	// therefore address exactly those slices -- NOT omp_get_num_threads() here.
	//
	// If mult's team is LARGER than qcham's was, len[task][thread_id] reads past
	// the end of the vector: a garbage length, then garbage ih/jh pointers, then
	// wild writes into wb. If it is SMALLER, the high slices are silently never
	// applied and matrix elements go missing. Neither is detectable from the
	// output except as a wrong energy.
	//
	// The team sizes can legitimately differ: qcham records
	// omp_get_num_threads() from inside its own parallel region (and does so
	// racily, from every thread), while the runtime is free to hand out a
	// different team later -- and any caller that changes OMP_NUM_THREADS or
	// enters from a different nesting depth between the two calls breaks the
	// assumption outright.
	//
	// Bind the loop to the actual number of stored slices and drive it with an
	// `omp for` so the mapping no longer depends on the team size at all. Row
	// index sets are disjoint across slices (each bra determinant belongs to one
	// alpha string, qcham.h:91), so the wb writes stay race-free.
	const size_t nslice = len[task].size();
	// SBD_CHECK_DISJOINT=1: the `omp parallel for` below is race-free ONLY if the
	// row index sets ih[task][sl][*] are disjoint across sl. If two slices write
	// the same row, `wb[ih] += ...` is a read-modify-write race: updates are lost
	// nondeterministically, the applied operator is no longer symmetric, and the
	// Rayleigh matrix <v_j,Hv_k> stops matching <v_k,Hv_j>. That is exactly the
	// observed signature -- Hasym jumps from 1e-14 at OMP<=2 to 4.4e-9 at OMP=3,
	// then grows geometrically until E falls below the true ground state.
	// This verifies the assumption instead of trusting the comment.
	if (std::getenv("SBD_CHECK_DISJOINT") != nullptr) {
	  std::unordered_map<size_t,size_t> owner;   // row -> first slice that writes it
	  size_t nshared = 0, worst_row = 0;
	  for (size_t sl = 0; sl < nslice; sl++)
	    for (size_t k = 0; k < len[task][sl]; k++) {
	      const size_t r = ih[task][sl][k];
	      auto itf = owner.find(r);
	      if (itf == owner.end()) owner.emplace(r, sl);
	      else if (itf->second != sl) { ++nshared; worst_row = r; }
	    }
	  if (nshared) {
	    int wr = 0; MPI_Comm_rank(MPI_COMM_WORLD, &wr);
	    std::cerr << " sbd: ROWS NOT DISJOINT task=" << task
		      << " nslice=" << nslice << " shared_writes=" << nshared
		      << " e.g. row " << worst_row
		      << " -- the parallel wb[ih] += is a RACE (world rank "
		      << wr << ")" << std::endl;
	  }
	}
#pragma omp parallel for schedule(static)
	for(size_t sl = 0; sl < nslice; sl++) {
	  for(size_t k=0; k < len[task][sl]; k++) {
	    wb[ih[task][sl][k]] += hij[task][sl][k]
	      * twk[jh[task][sl][k]];
	  }
	}
	if( task != slide.size() - 1 ) {
	  int bslide = slide[task]-slide[task+1];
	  rwk.resize(twk.size());
	  std::memcpy(rwk.data(),twk.data(),twk.size()*sizeof(ElemT));
	  sbd::MpiSlide(rwk,twk,bslide,b_comm);
	}
      }
      // SBD_CHECK_MULT=1: verify wb has the same length on every rank of the
      // communicator BEFORE reducing over it, and that every stored row index is
      // in range. MpiAllreduce passes wb.size() as the MPI count, so a mismatch
      // is reported by MPI only as an opaque MPI_ERR_TRUNCATE on "MPI COMM n",
      // with no indication of which vector or which rank went wrong. This turns
      // that into a named failure. Two ints per matvec when enabled.
      if (_sbd_check_mult()) {
        // Row indices must address wb; a write past the end corrupts the heap
        // (including vector bookkeeping) and can itself change wb.size().
        size_t max_ih = 0;
        for (size_t task = 0; task < slide.size(); task++)
          for (size_t th = 0; th < len[task].size(); th++)
            for (size_t k = 0; k < len[task][th]; k++)
              if (ih[task][th][k] > max_ih) max_ih = ih[task][th][k];
        if (!ih.empty() && max_ih >= wb.size()) {
          int wr = 0; MPI_Comm_rank(MPI_COMM_WORLD, &wr);
          std::cerr << " sbd: ERROR mult: stored row index " << max_ih
                    << " >= wb.size() " << wb.size() << " on world rank " << wr
                    << " -- writes past the end of the output vector."
                    << std::endl;
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
        _sbd_check_same_size(wb.size(), t_comm, "t_comm", "wb");
        _sbd_check_same_size(wb.size(), h_comm, "h_comm", "wb");
      }
      sbd::MpiAllreduce(wb,MPI_SUM,t_comm);
      sbd::MpiAllreduce(wb,MPI_SUM,h_comm);
    }

  } // end namespace gdb

} // end namespace sbd

#endif
