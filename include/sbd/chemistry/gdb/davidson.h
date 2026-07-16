/**
@file sbd/chemistry/gdb/davidson.h
@brief davidson for general determinant basis
*/
#ifndef SBD_CHEMISTRY_GDB_DAVIDSON_H
#define SBD_CHEMISTRY_GDB_DAVIDSON_H

namespace sbd {
  namespace gdb {
    
    template <typename ElemT>
    void BasisInitVector(std::vector<ElemT> & w,
			 const std::vector<std::vector<size_t>> & det,
			 MPI_Comm h_comm,
			 MPI_Comm b_comm,
			 MPI_Comm t_comm,
			 int init) {
      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);

      w.resize(det.size());
      if( init == 0 ) {
	if( mpi_rank_b == 0 ) {
	  w[0] = ElemT(1.0);
	}
	MpiBcast(w,0,t_comm);
      } else if ( init == 1 ) {
	if( mpi_rank_t == 0 ) {
	  Randomize(w,b_comm,h_comm);
	}
	MpiBcast(w,0,t_comm);
      }
    }

    template <typename ElemT, typename RealT>
    void Davidson(const std::vector<ElemT> & hii,
		  const std::vector<std::vector<size_t*>> & ih,
		  const std::vector<std::vector<size_t*>> & jh,
		  const std::vector<std::vector<ElemT*>> & hij,
		  const std::vector<std::vector<size_t>> & len,
		  const std::vector<int> & slide,
		  std::vector<ElemT> & w,
		  MPI_Comm h_comm,
		  MPI_Comm b_comm,
		  MPI_Comm t_comm,
		  int max_iteration,
		  int num_block,
		  RealT eps) {
      
      RealT eps_reg = 1.0e-12;

      std::vector<std::vector<ElemT>> v(num_block,w);
      std::vector<std::vector<ElemT>> Hv(num_block,w);
      std::vector<ElemT> r(w);
      std::vector<ElemT> dii(hii);
      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
      int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
      
      ElemT * H = (ElemT *) calloc(num_block*num_block,sizeof(ElemT));
      ElemT * U = (ElemT *) calloc(num_block*num_block,sizeof(ElemT));
      RealT * E = (RealT *) malloc(num_block*sizeof(RealT));
      char jobz = 'V';
      char uplo = 'U';
      int nb = num_block;
      MPI_Datatype DataE = GetMpiType<RealT>::MpiT;
      MPI_Datatype DataH = GetMpiType<ElemT>::MpiT;
      
      GetTotalD(hii,dii,h_comm);
      
      bool do_continue = true;
      
      for(int it=0; it < max_iteration; it++) {
	
#pragma omp parallel for
	for(size_t is=0; is < w.size(); is++) {
	  v[0][is] = w[is];
	}	

	for(int ib=0; ib < nb; ib++) {
	  
#ifdef SBD_DEBUG_MULT
	  for(int rank_h=0; rank_h < mpi_size_h; rank_h++) {
	    for(int rank_b=0; rank_b < mpi_size_b; rank_b++) {
	      for(int rank_t=0; rank_t < mpi_size_t; rank_t++) {
		if( mpi_rank_h == rank_h &&
		    mpi_rank_b == rank_b &&
		    mpi_rank_t == rank_t ) {
		  std::cout << " " << make_timestamp()
			    << " sbd: davidson step "
			    << it << "," << ib
			    << ", wave function weight before applying H at rank ("
			    << mpi_rank_h << ","
			    << mpi_rank_b << ","
			    << mpi_rank_t << "):";
		  for(size_t is=0; is < std::min(static_cast<size_t>(2),v[ib].size()); is++) {
		    std::cout << (( is == 0 ) ? " " : ",")
			      << v[ib][is];
		  }
		  if( v[ib].size() > static_cast<size_t>(2) ) {
		    std::cout << ", ..., "
			      << v[ib][v[ib].size()-1];
		  }
		  std::cout << std::endl;
		}
		MPI_Barrier(t_comm);
	      }
	      MPI_Barrier(b_comm);
	    }
	    MPI_Barrier(h_comm);
	  }
#endif
	
	  Zero(Hv[ib]);
	  mult(hii,ih,jh,hij,len,slide,
	       v[ib],Hv[ib],h_comm,b_comm,t_comm);

#ifdef SBD_DEBUG_MULT
	  for(int rank_h=0; rank_h < mpi_size_h; rank_h++) {
	    for(int rank_b=0; rank_b < mpi_size_b; rank_b++) {
	      for(int rank_t=0; rank_t < mpi_size_t; rank_t++) {
		if( mpi_rank_h == rank_h &&
		    mpi_rank_b == rank_b &&
		    mpi_rank_t == rank_t ) {
		  std::cout << " " << make_timestamp()
			    << " sbd: davidson step "
			    << it << "," << ib
			    << " wave function weight after applying H at rank ("
			    << mpi_rank_h << ","
			    << mpi_rank_b << ","
			    << mpi_rank_t << "):";
		  for(size_t is=0; is < std::min(static_cast<size_t>(2),Hv[ib].size()); is++) {
		    std::cout << (( is == 0 ) ? " " : ",")
			      << Hv[ib][is];
		  }
		  if( v[ib].size() > static_cast<size_t>(2) ) {
		    std::cout << ", ..., "
			      << Hv[ib][Hv[ib].size()-1];
		  }
		  std::cout << std::endl;
		}
		MPI_Barrier(t_comm);
	      }
	      MPI_Barrier(b_comm);
	    }
	    MPI_Barrier(h_comm);
	  }
#endif
	  
	  for(int jb=0; jb <= ib; jb++) {
	    InnerProduct(v[jb],Hv[ib],H[jb+nb*ib],b_comm);
	    H[ib+nb*jb] = Conjugate(H[jb+nb*ib]);
	  }
	  for(int jb=0; jb <= ib; jb++) {
	    for(int kb=0; kb <= ib; kb++) {
	      U[jb+nb*kb] = H[jb+nb*kb];
	    }
	  }
	  
	  hp_numeric::MatHeev(jobz,uplo,ib+1,U,nb,E);
	  
	  ElemT x = U[0];
#pragma omp parallel for
	  for(size_t is=0; is < w.size(); is++) {
	    w[is] = v[0][is] * x;
	  }
	  x = ElemT(-1.0) * U[0];
#pragma omp parallel for
	  for(size_t is=0; is < w.size(); is++) {
	    r[is] = Hv[0][is] * x;
	  }
	  for(int kb=1; kb <= ib; kb++) {
	    x = U[kb];
#pragma omp parallel for
	    for(size_t is=0; is < w.size(); is++) {
	      w[is] += v[kb][is] * x;
	    }
	    x = ElemT(-1.0) * U[kb];
#pragma omp parallel for
	    for(size_t is=0; is < w.size(); is++) {
	      r[is] += Hv[kb][is] * x;
	    }
	  }
#pragma omp parallel for
	  for(size_t is=0; is < w.size(); is++) {
	    r[is] += E[0]*w[is];
	  }
	  
	  
	  // #ifdef SBD_FUAGKUPATCH
	  MpiAllreduce(w,MPI_SUM,t_comm);
	  MpiAllreduce(w,MPI_SUM,h_comm);
	  MpiAllreduce(r,MPI_SUM,t_comm);
	  MpiAllreduce(r,MPI_SUM,h_comm);
	  ElemT volp(1.0/(mpi_size_h*mpi_size_t));
#pragma	omp parallel for
	  for(size_t is=0; is < w.size(); is++) {
	    w[is] *= volp;
	  }
#pragma omp parallel for
	  for(size_t is=0; is < r.size(); is++) {
	    r[is] *= volp;
	  }
	  // #endif

#ifdef SBD_DEBUG_MULT
	  for(int rank_h=0; rank_h < mpi_size_h; rank_h++) {
	    for(int rank_b=0; rank_b < mpi_size_b; rank_b++) {
	      for(int rank_t=0; rank_t < mpi_size_t; rank_t++) {
		if( mpi_rank_h == rank_h &&
		    mpi_rank_b == rank_b &&
		    mpi_rank_t == rank_t ) {
		  std::cout << " " << make_timestamp()
			    << " sbd: davidson step "
			    << it << "," << ib
			    << " residual vector at rank ("
			    << mpi_rank_h << ","
			    << mpi_rank_b << ","
			    << mpi_rank_t << "):";
		  for(size_t is=0; is < std::min(static_cast<size_t>(2),r.size()); is++) {
		    std::cout << (( is == 0 ) ? " " : ",")
			      << r[is];
		  }
		  if( v[ib].size() > static_cast<size_t>(2) ) {
		    std::cout << ", ..., "
			      << r[r.size()-1];
		  }
		  std::cout << std::endl;
		}
		MPI_Barrier(t_comm);
	      }
	      MPI_Barrier(b_comm);
	    }
	    MPI_Barrier(h_comm);
	  }
#endif
	  
	  RealT norm_w;
	  Normalize(w,norm_w,b_comm);
	  
	  RealT norm_r;
	  Normalize(r,norm_r,b_comm);
	  
	  if( mpi_rank_h == 0 ) {
	    if( mpi_rank_t == 0 ) {
	      if( mpi_rank_b == 0 ) {
		std::cout << " Davidson iteration " << it << "." << ib
			  << " (tol=" << norm_r << "):";
		for(int p=0; p < std::min(ib+1,4); p++) {
		  std::cout << " " << E[p];
		}
		std::cout << std::endl;
	      }	
	    }
	  }
	  
	  if( norm_r < eps ) {
	    do_continue = false;
	    break;
	  }
	  
	  if( ib < nb-1 ) {
	    // Determine
#pragma omp parallel for
	    for(size_t is=0; is < w.size(); is++) {
	      if( std::abs(E[0]-dii[is]) > eps_reg ) {
		v[ib+1][is] = r[is]/(E[0] - dii[is]);
	      } else {
		v[ib+1][is] = r[is]/(E[0] - dii[is] - eps_reg);
	      }
	    }
	    
	    // Gram-Schmidt orthogonalization
	    for(int kb=0; kb < ib+1; kb++) {
	      ElemT olap;
	      InnerProduct(v[kb],v[ib+1],olap,b_comm);
	      olap *= ElemT(-1.0);
#pragma omp parallel for
	      for(size_t is=0; is < w.size(); is++) {
		v[ib+1][is] += v[kb][is]*olap;
	      }
	    }
	    
	    RealT norm_v;
	    Normalize(v[ib+1],norm_v,b_comm);
	    
	  }
	} // end for(int ib=0; ib < nb; ib++)
	
	if( !do_continue ) {
	  break;
	}
	
	// Restart with C[0] = W;
#pragma omp parallel for
	for(size_t is=0; is < w.size(); is++) {
	  v[0][is] = w[is];
	}
	
      } // end for(int it=0; it < max_iteration; it++)
      
      free(H);
      free(U);
      free(E);
      
    }
    
    template <typename ElemT, typename RealT>
    void Davidson(const std::vector<ElemT> & hii,
		  std::vector<ElemT> & w,
		  const std::vector<std::vector<size_t>> & det,
		  const size_t bit_length,
		  const size_t norb,
		  const DetIndexMap & idxmap,
		  const std::vector<ExcitationLookup> & exidx,
		  const ElemT & I0,
		  const oneInt<ElemT> & I1,
		  const twoInt<ElemT> & I2,
		  MPI_Comm h_comm,
		  MPI_Comm b_comm,
		  MPI_Comm t_comm,
		  int max_iteration,
		  int num_block,
		  RealT eps) {
      
      RealT eps_reg = 1.0e-12;

      std::vector<std::vector<ElemT>> v(num_block,w);
      std::vector<std::vector<ElemT>> Hv(num_block,w);
      std::vector<ElemT> r(w);
      std::vector<ElemT> dii(hii);
      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
      int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
      
      ElemT * H = (ElemT *) calloc(num_block*num_block,sizeof(ElemT));
      ElemT * U = (ElemT *) calloc(num_block*num_block,sizeof(ElemT));
      RealT * E = (RealT *) malloc(num_block*sizeof(RealT));
      char jobz = 'V';
      char uplo = 'U';
      int nb = num_block;
      MPI_Datatype DataE = GetMpiType<RealT>::MpiT;
      MPI_Datatype DataH = GetMpiType<ElemT>::MpiT;
      
      GetTotalD(hii,dii,h_comm);

      bool do_continue = true;
      
      for(int it=0; it < max_iteration; it++) {
	
#pragma omp parallel for
	for(size_t is=0; is < w.size(); is++) {
	  v[0][is] = w[is];
	}
	
	for(int ib=0; ib < nb; ib++) {

#ifdef SBD_DEBUG_MULT
	  for(int rank_h=0; rank_h < mpi_size_h; rank_h++) {
	    for(int rank_b=0; rank_b < mpi_size_b; rank_b++) {
	      for(int rank_t=0; rank_t < mpi_size_t; rank_t++) {
		if( mpi_rank_h == rank_h &&
		    mpi_rank_b == rank_b &&
		    mpi_rank_t == rank_t ) {
		  std::cout << " " << make_timestamp()
			    << " sbd: davidson step "
			    << it << "," << ib
			    << ", wave function weight before applying H at rank ("
			    << mpi_rank_h << ","
			    << mpi_rank_b << ","
			    << mpi_rank_t << "):";
		  for(size_t is=0; is < static_cast<size_t>(2); is++) {
		    std::cout << (( is == 0 ) ? " " : ",")
			      << v[ib][is];
		  }
		  if( mpi_size_b == 1 ) {
		    std::cout << ", ...";
		    for(size_t is=v[ib].size()/2-2; is < v[ib].size()/2+2; is++) {
		      std::cout << "," << v[ib][is];
		    }
		  }
		  std::cout << ", ...";
		  for(size_t is=v[ib].size()-2; is < v[ib].size(); is++) {
		    std::cout << "," << v[ib][is];
		  }
		  std::cout << std::endl;
		}
		MPI_Barrier(t_comm);
	      }
	      MPI_Barrier(b_comm);
	    }
	    MPI_Barrier(h_comm);
	  }
#endif
	  
	  Zero(Hv[ib]);
	  mult(hii,v[ib],Hv[ib],bit_length,norb,
	       det,idxmap,exidx,I0,I1,I2,
	       h_comm,b_comm,t_comm);

#ifdef SBD_DEBUG_MULT
	  for(int rank_h=0; rank_h < mpi_size_h; rank_h++) {
	    for(int rank_b=0; rank_b < mpi_size_b; rank_b++) {
	      for(int rank_t=0; rank_t < mpi_size_t; rank_t++) {
		if( mpi_rank_h == rank_h &&
		    mpi_rank_b == rank_b &&
		    mpi_rank_t == rank_t ) {
		  std::cout << " " << make_timestamp()
			    << " sbd: davidson step "
			    << it << "," << ib
			    << " wave function weight after applying H at rank ("
			    << mpi_rank_h << ","
			    << mpi_rank_b << ","
			    << mpi_rank_t << "):";
		  for(size_t is=0; is < static_cast<size_t>(2); is++) {
		    std::cout << (( is == 0 ) ? " " : ",")
			      << Hv[ib][is];
		  }
		  if( mpi_size_b == 1 ) {
		    std::cout << ", ...";
		    for(size_t is=Hv[ib].size()/2-2; is < Hv[ib].size()/2+2; is++) {
		      std::cout << "," << Hv[ib][is];
		    }
		  }
		  std::cout << ", ...";
		  for(size_t is=Hv[ib].size()-2; is < Hv[ib].size(); is++) {
		    std::cout << "," << Hv[ib][is];
		  }
		  std::cout << std::endl;
		}
		MPI_Barrier(t_comm);
	      }
	      MPI_Barrier(b_comm);
	    }
	    MPI_Barrier(h_comm);
	  }
#endif
	  
	  
	  for(int jb=0; jb <= ib; jb++) {
	    InnerProduct(v[jb],Hv[ib],H[jb+nb*ib],b_comm);
	    H[ib+nb*jb] = Conjugate(H[jb+nb*ib]);
	  }
	  for(int jb=0; jb <= ib; jb++) {
	    for(int kb=0; kb <= ib; kb++) {
	      U[jb+nb*kb] = H[jb+nb*kb];
	    }
	  }

#ifdef SBD_DEBUG_MULT
	  if( mpi_rank_h == 0 &&
	      mpi_rank_b == 0 &&
	      mpi_rank_t == 0 ) {
	    std::cout << " " << make_timestamp()
		      << " sbd: davidson step "
		      << it << "," << ib
		      << " effective matrix = [";
	    for(int kb=0; kb <= ib; kb++) {
	      for(int jb=0; jb <= ib; jb++) {
		std::cout << ( (jb==0) ? "[" : "," ) << U[jb+nb*kb];
	      }
	      std::cout << "]";
	    }
	    std::cout << "]" << std::endl;
	  }
	      
#endif
	  
	  hp_numeric::MatHeev(jobz,uplo,ib+1,U,nb,E);
	  ElemT x = U[0];
#pragma omp parallel for
	  for(size_t is=0; is < w.size(); is++) {
	    w[is] = v[0][is] * x;
	  }
	  x = ElemT(-1.0) * U[0];
#pragma omp parallel for
	  for(size_t is=0; is < w.size(); is++) {
	    r[is] = Hv[0][is] * x;
	  }
	  for(int kb=1; kb <= ib; kb++) {
	    x = U[kb];
#pragma omp parallel for
	    for(size_t is=0; is < w.size(); is++) {
	      w[is] += v[kb][is] * x;
	    }
	    x = ElemT(-1.0) * U[kb];
#pragma omp parallel for
	    for(size_t is=0; is < w.size(); is++) {
	      r[is] += Hv[kb][is] * x;
	    }
	  }
#pragma omp parallel for
	  for(size_t is=0; is < w.size(); is++) {
	    r[is] += E[0]*w[is];
	  }
	  
	  MpiAllreduce(w,MPI_SUM,t_comm);
	  MpiAllreduce(w,MPI_SUM,h_comm);
	  MpiAllreduce(r,MPI_SUM,t_comm);
	  MpiAllreduce(r,MPI_SUM,h_comm);
	  ElemT volp(1.0/(mpi_size_h*mpi_size_t));
#pragma	omp parallel for
	  for(size_t is=0; is < w.size(); is++) {
	    w[is] *= volp;
	  }
#pragma omp parallel for
	  for(size_t is=0; is < r.size(); is++) {
	    r[is] *= volp;
	  }

#ifdef SBD_DEBUG_MULT
	  for(int rank_h=0; rank_h < mpi_size_h; rank_h++) {
	    for(int rank_b=0; rank_b < mpi_size_b; rank_b++) {
	      for(int rank_t=0; rank_t < mpi_size_t; rank_t++) {
		if( mpi_rank_h == rank_h &&
		    mpi_rank_b == rank_b &&
		    mpi_rank_t == rank_t ) {
		  std::cout << " " << make_timestamp()
			    << " sbd: davidson step "
			    << it << "," << ib
			    << " residual vector at rank ("
			    << mpi_rank_h << ","
			    << mpi_rank_b << ","
			    << mpi_rank_t << "):";
		  for(size_t is=0; is < static_cast<size_t>(2); is++) {
		    std::cout << (( is == 0 ) ? " " : ",")
			      << r[is];
		  }
		  if( mpi_size_b == 1 ) {
		    std::cout << ", ...";
		    for(size_t is=r.size()/2-2; is < r.size()/2+2; is++) {
		      std::cout << "," << r[is];
		    }
		  }
		  std::cout << ", ...";
		  for(size_t is=r.size()-2; is < r.size(); is++) {
		    std::cout << "," << r[is];
		  }
		  std::cout << std::endl;
		}
		MPI_Barrier(t_comm);
	      }
	      MPI_Barrier(b_comm);
	    }
	    MPI_Barrier(h_comm);
	  }
#endif
	  
	  RealT norm_w;
	  Normalize(w,norm_w,b_comm);
	  
	  RealT norm_r;
	  Normalize(r,norm_r,b_comm);
	  
	  if( mpi_rank_h == 0 ) {
	    if( mpi_rank_t == 0 ) {
	      if( mpi_rank_b == 0 ) {
		std::cout << " Davidson iteration " << it << "." << ib
			  << " (tol=" << norm_r << "):";
		for(int p=0; p < std::min(ib+1,4); p++) {
		  std::cout << " " << E[p];
		}
		std::cout << std::endl;
	      }	
	    }
	  }
	  if( norm_r < eps ) {
	    do_continue = false;
	    break;
	  }
	  if( ib < nb-1 ) {
	    // Determine
#pragma omp parallel for
	    for(size_t is=0; is < w.size(); is++) {
	      if( std::abs(E[0]-dii[is]) > eps_reg ) {
		v[ib+1][is] = r[is]/(E[0] - dii[is]);
	      } else {
		v[ib+1][is] = r[is]/(E[0] - dii[is] - eps_reg);
	      }
	    }
	    // Gram-Schmidt orthogonalization
	    for(int kb=0; kb < ib+1; kb++) {
	      ElemT olap;
	      InnerProduct(v[kb],v[ib+1],olap,b_comm);
	      olap *= ElemT(-1.0);
#pragma omp parallel for
	      for(size_t is=0; is < w.size(); is++) {
		v[ib+1][is] += v[kb][is]*olap;
	      }
	    }
	    RealT norm_v;
	    Normalize(v[ib+1],norm_v,b_comm);
	  }
	} // end for(int ib=0; ib < nb; ib++)
	if( !do_continue ) {
	  break;
	}
	// Restart with C[0] = W;
#pragma omp parallel for
	for(size_t is=0; is < w.size(); is++) {
	  v[0][is] = w[is];
	}
      } // end for(int it=0; it < max_iteration; it++)
      free(H);
      free(U);
      free(E);
    }

    /**
       Multi-root (block Davidson-Liu) diagonalization for the general
       determinant basis. Converges the `nroot` lowest eigenpairs.

       Additive twin of the self-contained single-root Davidson above; the two
       original single-root Davidson functions are left untouched. With
       nroot==1 this reproduces the single-root math and can serve as a
       regression oracle.

       @param[out] W     nroot eigenvectors (each length det.size()); seeded on input
       @param[out] Eout  nroot lowest energies (ascending)
       nb (subspace cap) is auto-bumped to >= nroot + max(nroot,10) and clamped to det.size().
       Convergence: absolute ||residual_p|| < eps for ALL roots.
    */
    template <typename ElemT, typename RealT>
    void DavidsonMultiRoot(const std::vector<ElemT> & hii,
			   std::vector<std::vector<ElemT>> & W,
			   std::vector<RealT> & Eout,
			   const std::vector<std::vector<size_t>> & det,
			   const size_t bit_length,
			   const size_t norb,
			   const DetIndexMap & idxmap,
			   const std::vector<ExcitationLookup> & exidx,
			   const ElemT & I0,
			   const oneInt<ElemT> & I1,
			   const twoInt<ElemT> & I2,
			   MPI_Comm h_comm,
			   MPI_Comm b_comm,
			   MPI_Comm t_comm,
			   int max_iteration,
			   int num_block,
			   int nroot,
			   RealT eps) {

      RealT eps_reg  = 1.0e-12;
      RealT tau_drop = 1.0e-6;   // relative linear-dependence drop for corrections

      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
      int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);

      // ---- subspace sizing: nb >= nroot + max(nroot,10), clamp to det.size() ----
      size_t local_len = det.size();
      int nb = num_block;
      int nb_min = nroot + std::max(nroot,10);
      if( nb < nb_min ) nb = nb_min;
      // Global dimension bound: the subspace cannot exceed the total number of
      // determinants. det.size() is the LOCAL chunk; the global count is the sum
      // over the b_comm partition. Bound nb by the global length.
      {
	long long ll = static_cast<long long>(local_len);
	long long gl = ll;
	MPI_Allreduce(&ll,&gl,1,MPI_LONG_LONG,MPI_SUM,b_comm);
	if( static_cast<long long>(nb) > gl ) nb = static_cast<int>(gl);
      }
      if( nb < nroot ) nb = nroot;
      if( num_block != nb && mpi_rank_h==0 && mpi_rank_t==0 && mpi_rank_b==0 ) {
	std::cout << " DavidsonMultiRoot: subspace size set to " << nb
		  << " (nroot=" << nroot << ")" << std::endl;
      }

      // ---- state ----
      std::vector<std::vector<ElemT>> v(nb, std::vector<ElemT>(local_len));
      std::vector<std::vector<ElemT>> Hv(nb, std::vector<ElemT>(local_len));
      std::vector<std::vector<ElemT>> Ritz(nroot, std::vector<ElemT>(local_len));
      std::vector<ElemT> res(local_len);
      std::vector<RealT> norm_r(nroot, RealT(0));
      std::vector<ElemT> dii(hii);
      GetTotalD(hii,dii,h_comm);

      ElemT * H = (ElemT *) calloc((size_t)nb*nb,sizeof(ElemT));
      ElemT * U = (ElemT *) calloc((size_t)nb*nb,sizeof(ElemT));
      RealT * E = (RealT *) malloc((size_t)nb*sizeof(RealT));
      char jobz = 'V';
      char uplo = 'U';
      ElemT volp(1.0/(mpi_size_h*mpi_size_t));

      // seed subspace with the nroot input guesses (assumed orthonormal-ish;
      // orthonormalize the set once to be safe)
      int nseed = nroot;
      for(int p=0; p < nroot; p++) v[p] = W[p];
      for(int p=0; p < nroot; p++) {
	for(int q=0; q < p; q++) {
	  ElemT olap; InnerProduct(v[q],v[p],olap,b_comm);
	  olap *= ElemT(-1.0);
#pragma omp parallel for
	  for(size_t is=0; is < local_len; is++) v[p][is] += v[q][is]*olap;
	}
	RealT nrm; Normalize(v[p],nrm,b_comm);
      }

      bool do_continue = true;

      for(int it=0; it < max_iteration && do_continue; it++) {

	int ib = 0;            // index of last basis vector in use
	int m  = 0;            // first vector not yet multiplied by H
	int ncur = nseed;      // number of basis vectors currently seeded

	while(true) {
	  // ---- (A) apply H to newly added basis vectors v[m .. ncur-1] ----
	  for(int jb = m; jb < ncur; jb++) {
	    Zero(Hv[jb]);
	    mult(hii,v[jb],Hv[jb],bit_length,norb,
		 det,idxmap,exidx,I0,I1,I2,
		 h_comm,b_comm,t_comm);
	  }
	  m  = ncur;
	  ib = ncur - 1;

	  // ---- (B) extend Hermitian Rayleigh matrix, copy to U ----
	  for(int jb=0; jb <= ib; jb++) {
	    for(int kb=0; kb <= ib; kb++) {
	      InnerProduct(v[jb],Hv[kb],H[jb+nb*kb],b_comm);
	    }
	  }
	  for(int jb=0; jb <= ib; jb++)
	    for(int kb=0; kb <= ib; kb++)
	      U[jb+nb*kb] = H[jb+nb*kb];

	  // ---- (C) diagonalize leading (ib+1) block ----
	  hp_numeric::MatHeev(jobz,uplo,ib+1,U,nb,E);

	  // ---- (D) per-root Ritz vector, residual, convergence ----
	  bool all_converged = true;
	  std::vector<int> unconverged;
	  for(int p=0; p < nroot; p++) {
	    ElemT x = U[0+nb*p];
#pragma omp parallel for
	    for(size_t is=0; is < local_len; is++) Ritz[p][is] = v[0][is]*x;
	    x = ElemT(-1.0) * U[0+nb*p];
#pragma omp parallel for
	    for(size_t is=0; is < local_len; is++) res[is] = Hv[0][is]*x;
	    for(int kb=1; kb <= ib; kb++) {
	      x = U[kb+nb*p];
#pragma omp parallel for
	      for(size_t is=0; is < local_len; is++) Ritz[p][is] += v[kb][is]*x;
	      x = ElemT(-1.0) * U[kb+nb*p];
#pragma omp parallel for
	      for(size_t is=0; is < local_len; is++) res[is] += Hv[kb][is]*x;
	    }
#pragma omp parallel for
	    for(size_t is=0; is < local_len; is++) res[is] += E[p]*Ritz[p][is];

	    // identical MPI reduce/scale per root
	    MpiAllreduce(Ritz[p],MPI_SUM,t_comm);
	    MpiAllreduce(Ritz[p],MPI_SUM,h_comm);
	    MpiAllreduce(res,MPI_SUM,t_comm);
	    MpiAllreduce(res,MPI_SUM,h_comm);
#pragma omp parallel for
	    for(size_t is=0; is < local_len; is++) Ritz[p][is] *= volp;
#pragma omp parallel for
	    for(size_t is=0; is < local_len; is++) res[is] *= volp;

	    RealT nrmw; Normalize(Ritz[p],nrmw,b_comm);
	    Normalize(res,norm_r[p],b_comm);

	    if( norm_r[p] >= eps ) { all_converged = false; unconverged.push_back(p); }
	  }

	  if( mpi_rank_h==0 && mpi_rank_t==0 && mpi_rank_b==0 ) {
	    RealT maxr = RealT(0);
	    for(int p=0;p<nroot;p++) if(norm_r[p]>maxr) maxr=norm_r[p];
	    std::cout << " DavidsonMR iteration " << it << "." << ib
		      << " (max_res=" << maxr << "):";
	    for(int p=0; p < nroot; p++) std::cout << " " << E[p];
	    std::cout << std::endl;
	  }

	  if( all_converged ) { do_continue = false; break; }

	  // ---- (E)+(F) build+append preconditioned corrections for unconverged roots ----
	  int appended = 0;
	  for(size_t u=0; u < unconverged.size(); u++) {
	    if( ib + 1 + appended >= nb ) break;   // subspace full -> collapse
	    int p = unconverged[u];
	    // rebuild this root's Ritz vector + residual (res buffer was reused across roots)
	    ElemT x0 = U[0+nb*p];
#pragma omp parallel for
	    for(size_t is=0; is < local_len; is++) Ritz[p][is] = v[0][is]*x0;
#pragma omp parallel for
	    for(size_t is=0; is < local_len; is++) res[is] = Hv[0][is]*(ElemT(-1.0)*x0);
	    for(int kb=1; kb <= ib; kb++) {
	      ElemT xk = U[kb+nb*p];
#pragma omp parallel for
	      for(size_t is=0; is < local_len; is++) Ritz[p][is] += v[kb][is]*xk;
	      ElemT xkn = ElemT(-1.0)*U[kb+nb*p];
#pragma omp parallel for
	      for(size_t is=0; is < local_len; is++) res[is] += Hv[kb][is]*xkn;
	    }
#pragma omp parallel for
	    for(size_t is=0; is < local_len; is++) res[is] += E[p]*Ritz[p][is];
	    MpiAllreduce(Ritz[p],MPI_SUM,t_comm); MpiAllreduce(Ritz[p],MPI_SUM,h_comm);
	    MpiAllreduce(res,MPI_SUM,t_comm);     MpiAllreduce(res,MPI_SUM,h_comm);
#pragma omp parallel for
	    for(size_t is=0; is < local_len; is++) Ritz[p][is] *= volp;
#pragma omp parallel for
	    for(size_t is=0; is < local_len; is++) res[is] *= volp;
	    RealT nrw; Normalize(Ritz[p],nrw,b_comm);
	    RealT nrr; Normalize(res,nrr,b_comm);

	    // preconditioned correction (per-root E[p])
	    int slot = ib + 1 + appended;
#pragma omp parallel for
	    for(size_t is=0; is < local_len; is++) {
	      if( std::abs(E[p]-dii[is]) > eps_reg )
		v[slot][is] = res[is]/(E[p] - dii[is]);
	      else
		v[slot][is] = res[is]/(E[p] - dii[is] - eps_reg);
	    }
	    // modified Gram-Schmidt against all current basis + accepted corrections,
	    // two passes (DGKS)
	    RealT norm_before = RealT(0);
	    for(int pass=0; pass < 2; pass++) {
	      for(int kb=0; kb < slot; kb++) {
		ElemT olap; InnerProduct(v[kb],v[slot],olap,b_comm);
		olap *= ElemT(-1.0);
#pragma omp parallel for
		for(size_t is=0; is < local_len; is++) v[slot][is] += v[kb][is]*olap;
	      }
	    }
	    RealT norm_v; Normalize(v[slot],norm_v,b_comm);
	    // linear-dependence drop: pre-normalization norm relative to unit input (1.0)
	    if( norm_v < tau_drop ) continue;
	    appended++;
	  }

	  if( appended == 0 ) break;   // nothing independent to add -> restart
	  ncur = ib + 1 + appended;    // new basis count; loop multiplies the new ones
	  if( ncur >= nb ) {
	    // subspace full: fall through to restart with Ritz seeds
	    break;
	  }
	} // end while (subspace growth)

	if( !do_continue ) break;

	// ---- (G) block restart: seed with the nroot Ritz vectors, re-orthonormalize ----
	for(int p=0; p < nroot; p++) v[p] = Ritz[p];
	for(int p=0; p < nroot; p++) {
	  for(int q=0; q < p; q++) {
	    ElemT olap; InnerProduct(v[q],v[p],olap,b_comm);
	    olap *= ElemT(-1.0);
#pragma omp parallel for
	    for(size_t is=0; is < local_len; is++) v[p][is] += v[q][is]*olap;
	  }
	  RealT nrm; Normalize(v[p],nrm,b_comm);
	}
	nseed = nroot;
      } // end restart loop

      // ---- output ----
      W.resize(nroot);
      Eout.resize(nroot);
      for(int p=0; p < nroot; p++) { W[p] = Ritz[p]; Eout[p] = E[p]; }

      free(H);
      free(U);
      free(E);
    }

  } // end namespace gdb
} // end namespace sbd

#endif
