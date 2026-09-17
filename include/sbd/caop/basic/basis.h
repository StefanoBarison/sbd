/**
@file sbd/caop/basic/basis.h
@brief function to setup the basis
 */
#ifndef SBD_CAOP_BASIC_BASIS_H
#define SBD_CAOP_BASIC_BASIS_H

#include <sys/stat.h>
#include <iomanip>
#include <cstdint>   // std::uint64_t below; do not rely on type_def.h to supply it

#include "sbd/framework/type_def.h"
#include "sbd/framework/mpi_utility.h"
#include "sbd/framework/bit_manipulation.h"

namespace sbd {
  
  template <typename Container>
  void redistribution(Container & config,
		      size_t bit_length,
		      size_t total_bit_length,
		      MPI_Comm comm) {
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    Container config_begin(mpi_size);
    Container config_end(mpi_size);
    std::vector<size_t> index_begin(mpi_size);
    std::vector<size_t> index_end(mpi_size);
    mpi_redistribution(config,config_begin,config_end,index_begin,index_end,
		       total_bit_length,bit_length,comm);
    
  }
  


  // Sort idx[lo..hi) so that (a[idx[i]][elem] & Mask) is in ascending order,
  // recursing on equal-valued runs through decreasing elements.
  // Mask is a template parameter so the compiler specialises each instantiation:
  // Mask=~0 eliminates the AND entirely; Mask=0x5555... is baked into code.
  template<size_t Mask = ~size_t(0), typename Container>
  void idx_sort_from_back(std::vector<size_t>& idx,
                          const Container& a,
                          size_t lo, size_t hi, int elem) {
    if (hi - lo <= 1 || elem < 0) return;
    std::sort(idx.begin() + lo, idx.begin() + hi,
              [&a, elem](size_t x, size_t y) {
                return (a[x][elem] & Mask) < (a[y][elem] & Mask);
              });
    if (elem == 0) return;
    size_t run_lo = lo;
    for (size_t i = lo + 1; i <= hi; i++) {
      if (i == hi || (a[idx[i]][elem] & Mask) != (a[idx[run_lo]][elem] & Mask)) {
        if (i - run_lo > 1)
          idx_sort_from_back<Mask>(idx, a, run_lo, i, elem - 1);
        run_lo = i;
      }
    }
  }

  // Equal-bra_a redistribution: partition dets so each rank owns a disjoint
  // contiguous range of alpha strings, giving equal bra_a across ranks.
  // Alpha occupation is at even bit positions (0,2,4,...) of the det bitstring,
  // interleaved with beta at odd positions.  Works for any clen.
  void redistribution_equal_bra_a(det_vector<size_t> & config,
                                   size_t bit_length,
                                   size_t total_bit_length,
                                   MPI_Comm comm) {
    int mpi_size; MPI_Comm_size(comm, &mpi_size);
    int mpi_rank; MPI_Comm_rank(comm, &mpi_rank);

    const int clen = static_cast<int>((total_bit_length + bit_length - 1) / bit_length);
    // Alpha bits sit at even positions (0,2,4,...) of each interleaved word.
    // Masking with ALPHA_MASK zeroes beta bits in place; the remaining even-position
    // bits compare correctly with < because bit position order is preserved.
    constexpr size_t ALPHA_MASK = 0x5555555555555555ULL;

    // Step 1: sort local dets alpha-primary by masking beta bits out of each word.
    // Det order within each alpha group is irrelevant; Step 7 re-sorts beta-primary.
    std::vector<size_t> idx(config.size());
    std::iota(idx.begin(), idx.end(), size_t(0));
    idx_sort_from_back<ALPHA_MASK>(idx, config, 0, idx.size(), clen - 1);
    {
      det_vector<size_t> tmp(config.size());
      for (size_t i = 0; i < config.size(); i++) tmp[i] = config[idx[i]];
      config = std::move(tmp);
    }

    // Step 2: collect local unique alpha keys from sorted config (mask on the fly).
    det_vector<size_t> local_alphas;
    {
      std::vector<size_t> prev(clen, ~size_t(0));
      for (size_t j = 0; j < config.size(); j++) {
        bool diff = false;
        for (int k = 0; k < clen && !diff; k++)
          diff = ((config[j][k] & ALPHA_MASK) != prev[k]);
        if (diff) {
          for (int k = 0; k < clen; k++)
            prev[k] = config[j][k] & ALPHA_MASK;
          local_alphas.emplace_back(prev);
        }
      }
    }

    // Step 3: allgather unique alpha keys → global sorted unique list.
    // Each alpha key is clen words; exchange as flat size_t arrays.
    int local_n = static_cast<int>(local_alphas.size());
    std::vector<int> all_counts(mpi_size);
    MPI_Allgather(&local_n, 1, MPI_INT, all_counts.data(), 1, MPI_INT, comm);
    int total_n = 0;
    std::vector<int> ag_displs_w(mpi_size, 0), ag_counts_w(mpi_size);
    for (int r = 0; r < mpi_size; r++) {
      ag_displs_w[r] = total_n * clen;
      ag_counts_w[r] = all_counts[r] * clen;
      total_n += all_counts[r];
    }
    std::vector<size_t> flat_local(local_n * clen), flat_all(total_n * clen);
    for (int i = 0; i < local_n; i++)
      for (int k = 0; k < clen; k++) flat_local[i * clen + k] = local_alphas[i][k];
    MPI_Allgatherv(flat_local.data(), local_n * clen, SBD_MPI_SIZE_T,
                   flat_all.data(), ag_counts_w.data(), ag_displs_w.data(), SBD_MPI_SIZE_T, comm);
    det_vector<size_t> all_alphas(static_cast<size_t>(total_n));
    for (int i = 0; i < total_n; i++)
      for (int k = 0; k < clen; k++) all_alphas[i][k] = flat_all[i * clen + k];
    sort_bitarray(all_alphas);
    size_t global_bra_a = all_alphas.size();

    // Step 4: assign alpha strings to ranks with equal bra_a.
    size_t chunk = global_bra_a / static_cast<size_t>(mpi_size);
    size_t rem   = global_bra_a % static_cast<size_t>(mpi_size);
    std::vector<size_t> alpha_start(mpi_size + 1, 0);
    for (int r = 0; r < mpi_size; r++)
      alpha_start[r+1] = alpha_start[r] + chunk + (static_cast<size_t>(r) < rem ? 1 : 0);

    // Step 5: for each local det compute destination rank.
    // all_alphas entries are already masked; apply ALPHA_MASK to config[j] inline.
    std::vector<int> dest_per_det(config.size());
    std::vector<int> sendcounts(mpi_size, 0);
    for (size_t j = 0; j < config.size(); j++) {
      size_t pos = static_cast<size_t>(
        std::lower_bound(all_alphas.begin(), all_alphas.end(), config[j],
          [clen](const auto& alpha, const auto& det) {
            constexpr size_t ALPHA_MASK = 0x5555555555555555ULL;
            for (int k = clen - 1; k >= 0; k--) {
              if (alpha[k] < (det[k] & ALPHA_MASK)) return true;
              if (alpha[k] > (det[k] & ALPHA_MASK)) return false;
            }
            return false;
          })
        - all_alphas.begin());
      int dest = static_cast<int>(
        std::upper_bound(alpha_start.begin(), alpha_start.end(), pos)
        - alpha_start.begin()) - 1;
      if (dest < 0) dest = 0;
      if (dest >= mpi_size) dest = mpi_size - 1;
      dest_per_det[j] = dest;
      sendcounts[dest]++;
    }

    // Step 6: MPI_Alltoallv (clen words per det).
    std::vector<int> sendcounts_w(mpi_size), sdispls(mpi_size, 0);
    for (int r = 0; r < mpi_size; r++) sendcounts_w[r] = sendcounts[r] * clen;
    for (int r = 1; r < mpi_size; r++) sdispls[r] = sdispls[r-1] + sendcounts_w[r-1];
    std::vector<int> recvcounts_w(mpi_size), rdispls(mpi_size, 0);
    MPI_Alltoall(sendcounts_w.data(), 1, MPI_INT, recvcounts_w.data(), 1, MPI_INT, comm);
    for (int r = 1; r < mpi_size; r++) rdispls[r] = rdispls[r-1] + recvcounts_w[r-1];
    int total_recv_w = rdispls[mpi_size-1] + recvcounts_w[mpi_size-1];

    std::vector<size_t> sendbuf(config.size() * clen);
    {
      std::vector<int> fill(sdispls);
      for (size_t j = 0; j < config.size(); j++) {
        int d = dest_per_det[j];
        for (int k = 0; k < clen; k++) sendbuf[fill[d]++] = config[j][k];
      }
    }
    std::vector<size_t> recvbuf(total_recv_w);
    MPI_Alltoallv(sendbuf.data(), sendcounts_w.data(), sdispls.data(), SBD_MPI_SIZE_T,
                  recvbuf.data(), recvcounts_w.data(), rdispls.data(), SBD_MPI_SIZE_T, comm);

    // Step 7: unpack and restore kernel sort order (beta-primary = less_from_back).
    size_t n_recv = static_cast<size_t>(total_recv_w) / static_cast<size_t>(clen);
    config.resize(n_recv);
    for (size_t i = 0; i < n_recv; i++)
      for (int k = 0; k < clen; k++) config[i][k] = recvbuf[i * clen + k];
    sort_bitarray(config);
  }

  // Config-aligned redistribution: partition dets so that every determinant
  // sharing a SPATIAL CONFIGURATION lands on the same rank.
  //
  // Two determinants have the same spatial configuration when, for every
  // spatial orbital, they agree on whether it is empty / singly / doubly
  // occupied -- i.e. they differ only in the spin arrangement of their open
  // shells. Such a set is exactly one Sz-orbit, and it is exactly one block of
  // the single-spin projector V (see chemistry/gdb/single_spin.h). Keeping
  // whole orbits rank-local is what lets the projected (Option 2) solver run
  // with b_comm_size > 1 at all: a split orbit truncates a block, so its CSF
  // column is neither normalized nor spin-pure, and the per-rank CSF counts no
  // longer partition the global space that the MPI_Exscan numbering assumes.
  //
  // Key. With alpha at even bit positions and beta at odd positions, let
  //   a = w & ALPHA_MASK          (alpha bits, in place)
  //   b = (w >> 1) & ALPHA_MASK   (beta bits, shifted onto the even positions)
  // then (a|b, a&b) = (occupied-anywhere, doubly-occupied) determines the
  // configuration uniquely, and both halves are plain masked words -- the same
  // style as redistribution_equal_bra_a's ALPHA_MASK trick.
  //
  // Balance. Orbit sizes are C(n_open, n_up) and vary widely (1, 2, 6, 20, 70,
  // 252, ... for Sz=0), so slicing the key list into equal COUNTS of keys
  // would imbalance badly. Instead assign whole keys greedily largest-first to
  // the currently-least-loaded rank. Measured on real sampled N2 spaces this
  // gives <=0.6% imbalance up to 64 ranks, improving with system size.
  // The assignment is computed from the globally allgathered key list, which is
  // identical on every rank, so all ranks derive the same ownership without
  // further communication.
  //
  // Order. Like redistribution_equal_bra_a, this ends by restoring the
  // canonical less_from_back order locally (Step 7). That is REQUIRED, not
  // cosmetic: makeDetIndexMap builds AdetToBdetSM / BdetToAdetSM by push_back
  // in det-list order, and ~25 lower_bound calls in mult.h / qcham.h /
  // correlation.h binary-search those rows. A non-canonical local order makes
  // them silently miss matches, dropping Hamiltonian terms with no error.
  // LoadWavefunction also binary-searches the det list with less_from_back.
  void redistribution_equal_config(det_vector<size_t> & config,
                                   size_t bit_length,
                                   size_t total_bit_length,
                                   MPI_Comm comm) {
    int mpi_size; MPI_Comm_size(comm, &mpi_size);
    int mpi_rank; MPI_Comm_rank(comm, &mpi_rank);
    if (mpi_size == 1) { sort_bitarray(config); return; }

    const int clen = static_cast<int>((total_bit_length + bit_length - 1) / bit_length);
    constexpr size_t ALPHA_MASK = 0x5555555555555555ULL;

    // Config key of one det, as 2*clen words: [occupied-anywhere | doubly-occ].
    auto config_key = [clen](const auto & det, std::vector<size_t> & key) {
      for (int k = 0; k < clen; k++) {
        const size_t a = det[k] & ALPHA_MASK;
        const size_t b = (det[k] >> 1) & ALPHA_MASK;
        key[k]        = a | b;
        key[clen + k] = a & b;
      }
    };
    const int klen = 2 * clen;

    // Step 1: local (key, det index) list, sorted by key so equal keys are adjacent.
    std::vector<size_t> keybuf(static_cast<size_t>(config.size()) * klen);
    for (size_t j = 0; j < config.size(); j++) {
      std::vector<size_t> key(klen);
      config_key(config[j], key);
      for (int k = 0; k < klen; k++) keybuf[j * klen + k] = key[k];
    }
    auto key_less = [klen](const size_t * x, const size_t * y) {
      for (int k = klen - 1; k >= 0; k--) {   // from-back, matching less_from_back
        if (x[k] < y[k]) return true;
        if (x[k] > y[k]) return false;
      }
      return false;
    };
    std::vector<size_t> order(config.size());
    std::iota(order.begin(), order.end(), size_t(0));
    std::sort(order.begin(), order.end(), [&](size_t x, size_t y) {
      return key_less(&keybuf[x * klen], &keybuf[y * klen]);
    });

    // Step 2: local unique keys, with local multiplicity (orbit size on this rank).
    std::vector<size_t> loc_keys;      // flat, klen words per key
    std::vector<size_t> loc_weight;    // dets per key, local
    for (size_t i = 0; i < order.size(); i++) {
      const size_t * cur = &keybuf[order[i] * klen];
      if (i == 0 || key_less(&keybuf[order[i-1] * klen], cur)) {
        loc_keys.insert(loc_keys.end(), cur, cur + klen);
        loc_weight.push_back(1);
      } else {
        loc_weight.back()++;
      }
    }
    const int loc_n = static_cast<int>(loc_weight.size());

    // Step 3: allgather keys + weights -> global key list, identical on all ranks.
    std::vector<int> all_counts(mpi_size);
    MPI_Allgather(&loc_n, 1, MPI_INT, all_counts.data(), 1, MPI_INT, comm);
    int total_n = 0;
    std::vector<int> kdispl(mpi_size, 0), kcount(mpi_size);
    std::vector<int> wdispl(mpi_size, 0), wcount(mpi_size);
    for (int r = 0; r < mpi_size; r++) {
      kdispl[r] = total_n * klen;  kcount[r] = all_counts[r] * klen;
      wdispl[r] = total_n;         wcount[r] = all_counts[r];
      total_n += all_counts[r];
    }
    std::vector<size_t> gath_keys(static_cast<size_t>(total_n) * klen);
    std::vector<size_t> gath_weight(total_n);
    MPI_Allgatherv(loc_keys.data(), loc_n * klen, SBD_MPI_SIZE_T,
                   gath_keys.data(), kcount.data(), kdispl.data(), SBD_MPI_SIZE_T, comm);
    MPI_Allgatherv(loc_weight.data(), loc_n, SBD_MPI_SIZE_T,
                   gath_weight.data(), wcount.data(), wdispl.data(), SBD_MPI_SIZE_T, comm);

    // Step 4: merge duplicate keys (the same config can appear on several ranks),
    // producing the global unique key list with global orbit weights.
    std::vector<size_t> gorder(total_n);
    std::iota(gorder.begin(), gorder.end(), size_t(0));
    std::sort(gorder.begin(), gorder.end(), [&](size_t x, size_t y) {
      return key_less(&gath_keys[x * klen], &gath_keys[y * klen]);
    });
    std::vector<size_t> uniq_keys;     // flat, klen words per key
    std::vector<size_t> uniq_weight;
    for (size_t i = 0; i < gorder.size(); i++) {
      const size_t * cur = &gath_keys[gorder[i] * klen];
      if (i == 0 || key_less(&gath_keys[gorder[i-1] * klen], cur)) {
        uniq_keys.insert(uniq_keys.end(), cur, cur + klen);
        uniq_weight.push_back(gath_weight[gorder[i]]);
      } else {
        uniq_weight.back() += gath_weight[gorder[i]];
      }
    }
    const size_t n_uniq = uniq_weight.size();

    // Step 5: greedy largest-first assignment of whole keys to ranks. Ties are
    // broken by key order and then by lowest rank index, so the result is
    // deterministic and identical on every rank.
    std::vector<size_t> by_weight(n_uniq);
    std::iota(by_weight.begin(), by_weight.end(), size_t(0));
    std::stable_sort(by_weight.begin(), by_weight.end(),
                     [&](size_t x, size_t y) { return uniq_weight[x] > uniq_weight[y]; });
    std::vector<size_t> load(mpi_size, 0);
    std::vector<int> key_owner(n_uniq, 0);
    for (size_t t = 0; t < n_uniq; t++) {
      const size_t kk = by_weight[t];
      int best = 0;
      for (int r = 1; r < mpi_size; r++) if (load[r] < load[best]) best = r;
      key_owner[kk] = best;
      load[best] += uniq_weight[kk];
    }

    // Step 6: destination of each local det = owner of its config key.
    std::vector<int> dest_per_det(config.size());
    std::vector<int> sendcounts(mpi_size, 0);
    {
      std::vector<size_t> key(klen);
      for (size_t j = 0; j < config.size(); j++) {
        config_key(config[j], key);
        // binary search the unique key list (sorted by key_less)
        size_t lo = 0, hi = n_uniq;
        while (lo < hi) {
          const size_t mid = lo + (hi - lo) / 2;
          if (key_less(&uniq_keys[mid * klen], key.data())) lo = mid + 1;
          else hi = mid;
        }
        const int dest = (lo < n_uniq) ? key_owner[lo] : 0;
        dest_per_det[j] = dest;
        sendcounts[dest]++;
      }
    }

    // Step 7: MPI_Alltoallv (clen words per det).
    std::vector<int> sendcounts_w(mpi_size), sdispls(mpi_size, 0);
    for (int r = 0; r < mpi_size; r++) sendcounts_w[r] = sendcounts[r] * clen;
    for (int r = 1; r < mpi_size; r++) sdispls[r] = sdispls[r-1] + sendcounts_w[r-1];
    std::vector<int> recvcounts_w(mpi_size), rdispls(mpi_size, 0);
    MPI_Alltoall(sendcounts_w.data(), 1, MPI_INT, recvcounts_w.data(), 1, MPI_INT, comm);
    for (int r = 1; r < mpi_size; r++) rdispls[r] = rdispls[r-1] + recvcounts_w[r-1];
    int total_recv_w = rdispls[mpi_size-1] + recvcounts_w[mpi_size-1];

    std::vector<size_t> sendbuf(static_cast<size_t>(config.size()) * clen);
    {
      std::vector<int> fill(sdispls);
      for (size_t j = 0; j < config.size(); j++) {
        const int d = dest_per_det[j];
        for (int k = 0; k < clen; k++) sendbuf[fill[d]++] = config[j][k];
      }
    }
    std::vector<size_t> recvbuf(total_recv_w);
    MPI_Alltoallv(sendbuf.data(), sendcounts_w.data(), sdispls.data(), SBD_MPI_SIZE_T,
                  recvbuf.data(), recvcounts_w.data(), rdispls.data(), SBD_MPI_SIZE_T, comm);

    // Step 8: unpack and restore the canonical order (see the Order note above).
    // sort_bitarray also dedups, matching redistribution_equal_bra_a.
    size_t n_recv = static_cast<size_t>(total_recv_w) / static_cast<size_t>(clen);
    config.resize(n_recv);
    for (size_t i = 0; i < n_recv; i++)
      for (int k = 0; k < clen; k++) config[i][k] = recvbuf[i * clen + k];
    sort_bitarray(config);
  }

  template <typename Container>
  void reordering(Container & config,
		  size_t bit_length,
		  size_t total_bit_length,
		  MPI_Comm comm) {
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    Container config_begin(mpi_size);
    Container config_end(mpi_size);
    std::vector<size_t> index_begin(mpi_size);
    std::vector<size_t> index_end(mpi_size);
    mpi_sort_bitarray(config,config_begin,config_end,index_begin,index_end,
		      total_bit_length,bit_length,comm);
  }
  
  // I/O for basis
  template<typename Container>
  void load_basis_from_file(const std::string & filename,
			    Container & config,
			    size_t bit_length,
			    size_t total_bit_length) {
    if( get_extension(filename) == std::string("txt") ) {
      std::ifstream ifs(filename);
      if( !ifs.is_open() ) {
	throw std::runtime_error("Failed to open basis bit-string file.");
      }
      std::string line;
      std::vector<std::string> lines;
      while( std::getline(ifs,line) ) {
	lines.push_back(line);
      }
      config.resize(lines.size());
      for(size_t i=0; i < lines.size(); i++) {
	config[i] = from_string(lines[i],bit_length,total_bit_length);
      }
    } else if ( get_extension(filename) == std::string("bin") ) {
      std::ifstream ifs(filename, std::ios::binary);
      if( !ifs.is_open() ) {
	throw std::runtime_error("Failed to open basis bit-string binary file.");
      }

      size_t inner_size = (total_bit_length+bit_length-1)/bit_length;
      ifs.seekg(0, std::ios::end);
      std::streampos file_size = ifs.tellg();
      ifs.seekg(0, std::ios::beg);

      size_t bytes_per_line = inner_size * sizeof(size_t);

      if( file_size % bytes_per_line != 0 ) {
	throw std::runtime_error("Binary file size mismatch");
      }

      size_t num_lines = file_size / bytes_per_line;

      config.resize(num_lines);
      for(size_t i=0; i < num_lines; i++) {
	config[i].resize(inner_size);
	ifs.read(reinterpret_cast<char*>(config[i].data()),bytes_per_line);
	if (!ifs) {
	  throw std::runtime_error("Failed to read binary basis data.");
	}
      }
    }
  }
  
  template<typename Container>
  void save_basis_to_file(const std::string & filename,
			  Container & config,
			  size_t bit_length,
			  size_t total_bit_length) {
    if( get_extension(filename) == std::string("txt") ) {
      std::ofstream ofs(filename);
      for(size_t i=0; i < config.size(); i++) {
	ofs << makestring(config[i],bit_length,total_bit_length) << std::endl;
      }
    } else if ( get_extension(filename) == std::string("bin") ) {
      std::ofstream ofs(filename,std::ios::binary);
      for(auto & b : config) {
	ofs.write(reinterpret_cast<char*>(b.data()),sizeof(size_t)*b.size());
      }
    }
  }

  // basis file name for multiple nodes
  std::string basisfilename(const std::string & basisname, int index, int filetype) {
    std::ostringstream oss;
    oss << std::setw(6) << std::setfill('0') << index;
    std::string tag = oss.str();
    std::string filename;
    if( filetype == 0 ) {
      filename = basisname + tag + ".txt";
    } else if ( filetype == 1 ) {
      filename = basisname + tag + ".bin";
    }
    return filename;
  }

  template<typename Container>
  void load_basis_from_files(const std::vector<std::string> & all_filenames,
			     Container & config,
			     size_t bit_length,
			     size_t total_bit_length,
			     MPI_Comm comm) {
    int mpi_rank; MPI_Comm_rank(comm, &mpi_rank);
    int mpi_size; MPI_Comm_size(comm, &mpi_size);
    
    const int num_files = static_cast<int>(all_filenames.size());
    config.clear();
    
    if (num_files == 0) return;
    
    const int base = num_files / mpi_size;
    const int rem  = num_files % mpi_size;

    int my_first = 0;
    int my_count = 0;
    if (mpi_rank < rem) {
      my_count = base + 1;
      my_first = mpi_rank * my_count;
    } else {
      my_count = base;
      my_first = rem * (base + 1) + (mpi_rank - rem) * base;
    }
    const int my_last = my_first + my_count;
    
    for (int i = my_first; i < my_last; ++i) {
      const std::string & fname = all_filenames[i];
      
      Container local;
      load_basis_from_file(fname, local, bit_length, total_bit_length);
      
      config.insert(config.end(),
		    std::make_move_iterator(local.begin()),
		    std::make_move_iterator(local.end()));
    }
    sort_bitarray(config);
  }
  
  // load single file
  void load_basis_from_single_binary(const std::string & filename,
				     std::vector<std::vector<size_t>> & config,
				     size_t bit_length,
				     size_t total_bit_length,
				     MPI_Comm comm) {
    int mpi_rank; MPI_Comm_rank(comm, &mpi_rank);
    int mpi_size; MPI_Comm_size(comm, &mpi_size);
    
    const size_t inner_size    = (total_bit_length + bit_length - 1) / bit_length;
    const size_t bytes_per_line = inner_size * sizeof(size_t);
    
    std::uint64_t num_lines_u64 = 0;
    
    if (mpi_rank == 0) {
      std::ifstream ifs(filename, std::ios::binary);
      if (!ifs.is_open()) {
	throw std::runtime_error("Failed to open basis binary file: " + filename);
      }
      
      ifs.seekg(0, std::ios::end);
      std::streampos file_size_pos = ifs.tellg();
      ifs.seekg(0, std::ios::beg);
      
    if (file_size_pos < 0) {
      throw std::runtime_error("tellg() failed for file: " + filename);
    }
    
    const std::uint64_t file_size = static_cast<std::uint64_t>(file_size_pos);
    
    if (file_size % bytes_per_line != 0) {
      throw std::runtime_error("Binary file size mismatch in " + filename);
    }
    
    num_lines_u64 = file_size / bytes_per_line;
    }
    
    MPI_Bcast(&num_lines_u64, 1, MPI_UINT64_T, 0, comm);

    if (num_lines_u64 == 0) {
      config.clear();
      return;
    }
    
    const std::size_t num_lines = static_cast<std::size_t>(num_lines_u64);

    const std::size_t base = num_lines / mpi_size;
    const std::size_t rem  = num_lines % mpi_size;
    
    std::size_t my_first = 0;
    std::size_t my_count = 0;
    if (static_cast<std::size_t>(mpi_rank) < rem) {
      my_count = base + 1;
      my_first = static_cast<std::size_t>(mpi_rank) * my_count;
    } else {
      my_count = base;
      my_first = rem * (base + 1)
	+ (static_cast<std::size_t>(mpi_rank) - rem) * base;
    }
    const std::size_t my_last = my_first + my_count;
    
    config.clear();
    config.resize(my_count);
    for (auto & row : config) {
      row.resize(inner_size);
    }
    
    if (my_count == 0) {
      return;
    }
    
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
      throw std::runtime_error("Failed to open basis binary file (per rank): " + filename);
    }

    const std::uint64_t my_offset_bytes =
      static_cast<std::uint64_t>(my_first) * bytes_per_line;
    
    ifs.seekg(static_cast<std::streamoff>(my_offset_bytes), std::ios::beg);
    if (!ifs) {
      throw std::runtime_error("seekg failed for basis binary file: " + filename);
    }
    
    for (std::size_t i = 0; i < my_count; ++i) {
      ifs.read(reinterpret_cast<char*>(config[i].data()), bytes_per_line);
      if (!ifs) {
	throw std::runtime_error("Failed to read basis data from: " + filename);
      }
    }
    sort_bitarray(config);
  }

  inline void mpi_bcast_string_vector(std::vector<std::string> & vec,
				      int root,
				      MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    int count = static_cast<int>(vec.size());
    MPI_Bcast(&count, 1, MPI_INT, root, comm);
    
    if (rank != root) {
      vec.resize(count);
    }
    
    std::vector<int> lengths(count);
    if (rank == root) {
      for (int i = 0; i < count; i++) {
	lengths[i] = static_cast<int>(vec[i].size());
      }
    }
    MPI_Bcast(lengths.data(), count, MPI_INT, root, comm);
    
    for (int i = 0; i < count; i++) {
      if (rank != root) {
	vec[i].resize(lengths[i]);
      }
      if (lengths[i] > 0) {
#ifdef SBD_TRADMODE
	char * ptr = &vec[i][0];
	MPI_Bcast(ptr, lengths[i], MPI_CHAR, root, comm);
#else
	MPI_Bcast(vec[i].data(), lengths[i], MPI_CHAR, root, comm);
#endif
      }
    }
  }
  
  
}

#endif
