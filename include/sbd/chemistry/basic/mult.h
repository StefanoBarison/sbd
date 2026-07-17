/**
@file sbd/chemistry/base/mult.h
@brief Function to perform Hamiltonian operation for twist-basis parallelization scheme
*/
#ifndef SBD_CHEMISTRY_MULT_BASE_H
#define SBD_CHEMISTRY_MULT_BASE_H

#include <chrono>
#include <cstdio>

#ifdef SBD_USE_NCCL
#include <nccl.h>
#endif


namespace sbd
{

template <typename ElemT>
class MultBase {
protected:
    uint32_t bit_length_;
    size_t norbs_;
    size_t D_size_;      // the vector length of a full (i.e., alpha + beta) determinant
    size_t D_half_size_; // the vector length of a half (i.e., alpha or beta) determinant
    MPI_Comm h_comm_;
    MPI_Comm b_comm_;
    MPI_Comm t_comm_;
    MPI_Comm a_comm_;
#ifdef SBD_USE_NCCL
    ncclComm_t h_nccl_comm_;
    ncclComm_t b_nccl_comm_;
    ncclComm_t t_nccl_comm_;
    ncclComm_t a_nccl_comm_;
#endif
public:
    MultBase() : a_comm_(MPI_COMM_SELF) {}

    MultBase(uint32_t bit_length, size_t norbs, MPI_Comm h_comm, MPI_Comm b_comm, MPI_Comm t_comm)
        : bit_length_(bit_length), norbs_(norbs), h_comm_(h_comm), b_comm_(b_comm), t_comm_(t_comm),
          a_comm_(MPI_COMM_SELF)
    {
        D_size_ = (2 * norbs + bit_length - 1) / bit_length;
        D_half_size_ = (norbs + bit_length - 1) / bit_length;
    }

    inline uint32_t bit_length(void) const
    {
        return bit_length_;
    }
    inline size_t norbs(void) const
    {
        return norbs_;
    }
    inline size_t D_size(void) const
    {
        return D_size_;
    }
    inline MPI_Comm h_comm(void) const
    {
        return h_comm_;
    }
    inline MPI_Comm b_comm(void) const
    {
        return b_comm_;
    }
    inline MPI_Comm t_comm(void) const
    {
        return t_comm_;
    }
    inline MPI_Comm a_comm(void) const
    {
        return a_comm_;
    }
#ifdef SBD_USE_NCCL
    inline ncclComm_t h_nccl_comm(void) const
    {
        return h_nccl_comm_;
    }
    inline ncclComm_t b_nccl_comm(void) const
    {
        return b_nccl_comm_;
    }
    inline ncclComm_t t_nccl_comm(void) const
    {
        return t_nccl_comm_;
    }
    inline ncclComm_t a_nccl_comm(void) const
    {
        return a_nccl_comm_;
    }
#endif

#ifdef SBD_THRUST
    virtual void run(const thrust::device_vector<ElemT> &hii,
                    const thrust::device_vector<ElemT> &Wk,
                    thrust::device_vector<ElemT> &Wb) = 0;
#else
    virtual void run(const std::vector<ElemT> & hii,
            	    const std::vector<ElemT> & Wk,
	                std::vector<ElemT> & Wb) = 0;
#endif
};


}

#endif
