/**
@file sbd/chemistry/gdb/carryover.h
@brief functions for carryover bitstrings for gdb
*/
#ifndef SBD_CHEMISTRY_GDB_CARRYOVER_H
#define SBD_CHEMISTRY_GDB_CARRYOVER_H

namespace sbd {
  namespace gdb {

    template <typename ElemT, typename RealT, typename DetsContainer>
    void CarryOverDet(const std::vector<ElemT> & w,
		      const DetsContainer & det,
		      MPI_Comm b_comm,
		      size_t kept,
		      sbd::det_vector<size_t> & rdet,
		      RealT & discarted_weight) {
      // using RealT = typename GetRealType<ElemT>::RealT;
      std::vector<RealT> r(w.size());
#pragma omp parallel for
      for(size_t i=0; i < w.size(); i++) {
	r[i] = SquaredNorm(w[i]);
      }
      std::vector<size_t> ranking(r.size());
      mpi_find_ranking(r,ranking,b_comm);
      size_t rdet_size = 0;
      // Pin the team size on BOTH regions below with num_threads(nthr).
      //
      // These two regions must use the IDENTICAL strided partition: the first
      // counts per-thread hits, offset[] is a prefix sum over those counts, and
      // the second writes into rdet starting at offset[thread_id]. The stride is
      // `nthr`, so a team smaller than nthr in either region would cover
      // different residue classes than the offsets were computed for --
      // overlapping writes into rdet and a wrong rdet_size, silently.
      // omp_get_max_threads() is the size of the NEXT team, not a guarantee, and
      // dynamic adjustment is ON by default in libgomp. num_threads() makes both
      // teams exactly nthr, so the two partitions match by construction.
      int nthr = omp_get_max_threads();
      if( nthr < 1 ) nthr = 1;
      const size_t num_threads = static_cast<size_t>(nthr);
      std::vector<size_t> local_size(num_threads,0);
#pragma omp parallel num_threads(nthr)
      {
	size_t thread_id = omp_get_thread_num();
	for(size_t k=thread_id; k < det.size(); k+=num_threads) {
	  if( ranking[k] < kept ) {
	    local_size[thread_id]++;
	  }
	}
      }
      std::vector<size_t> offset(num_threads,0);
      for(size_t tid=0; tid < num_threads; tid++) {
	rdet_size += local_size[tid];
	if( tid < num_threads-1 ) {
	  offset[tid+1] = rdet_size;
	}
      }
      rdet.resize(rdet_size);
      std::vector<RealT> keep_weight_local(num_threads,0.0);
#pragma omp parallel num_threads(nthr)   // must match the counting region above
      {
	size_t thread_id = omp_get_thread_num();
	size_t local_addr = offset[thread_id];
	for(size_t k=thread_id; k < det.size(); k+=num_threads) {
	  if( ranking[k] < kept ) {
	    rdet[local_addr++] = det[k];
	    keep_weight_local[thread_id] += r[k];
	  }
	}
      }
      RealT keep_weight = 0.0;
      for(size_t tid=0; tid < num_threads; tid++) {
	keep_weight += keep_weight_local[tid];
      }
      RealT keep_weight_global = 0.0;
      MPI_Datatype DataT = GetMpiType<RealT>::MpiT;
      MPI_Allreduce(&keep_weight,&keep_weight_global,
		    1,DataT,MPI_SUM,b_comm);
      discarted_weight = 1.0-keep_weight_global;
      sort_bitarray(rdet);
    }

    template <typename ElemT, typename RealT, typename InDetsContainer, typename OutDetsContainer>
    void WeightTruncation(const std::vector<ElemT> & w,
			  const InDetsContainer & det,
			  RealT threshold,
			  std::vector<ElemT> & rw,
			  OutDetsContainer & rdet) {

      const size_t n = det.size();

      if( w.size() != n) {
	throw std::runtime_error("CarryoverDet: w.size() != det.size()");
      }

      rw.clear();
      rdet.clear();

      std::vector<size_t> keep(n,0);
      
#pragma omp parallel for
      for(size_t i=0; i < n; ++i) {
	const auto abs_w = std::abs(w[i]);
	const auto weight = abs_w * abs_w;
	if( threshold < weight ) {
	  keep[i] = 1;
	}
      }

      std::vector<size_t> offset(n+1,0);
      for(size_t i=0; i < n; ++i) {
	offset[i+1] = offset[i] + keep[i];
      }

      const size_t nr = offset[n];
      rw.resize(nr);
      rdet.resize(nr);
      
#pragma omp parallel for
      for(size_t i=0; i < n; ++i) {
	if(keep[i]) {
	  const size_t j=offset[i];
	  rw[j] = w[i];
	  rdet[j] = det[i];
	}
      }

    }

  } // end namespace gdb
} // end namespace sbd

#endif
