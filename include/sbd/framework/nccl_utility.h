/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SBD_FRAMEWORK_NCCL_UTILITY_H
#define SBD_FRAMEWORK_NCCL_UTILITY_H
#ifdef SBD_USE_NCCL

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <type_traits>

#include <mpi.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include "sbd/framework/nvtx.h"
#include "sbd/framework/cuda_utility.h"
#include <nccl.h>

// -----------------------------------------------------------------------------
// NCCL datatype mapping
// -----------------------------------------------------------------------------
template <typename T>
struct SbdNcclDataType;

template <>
struct SbdNcclDataType<float> {
    static constexpr ncclDataType_t value = ncclFloat;
};

template <>
struct SbdNcclDataType<double> {
    static constexpr ncclDataType_t value = ncclDouble;
};

template <>
struct SbdNcclDataType<int> {
    static constexpr ncclDataType_t value = ncclInt;
};

template <>
struct SbdNcclDataType<int64_t> {
    static constexpr ncclDataType_t value = ncclInt64;
};

// -----------------------------------------------------------------------------
// NCCL error check
// -----------------------------------------------------------------------------
#ifndef SBD_CHECK_NCCL
#define SBD_CHECK_NCCL(cmd)                                              \
    do {                                                                 \
        ncclResult_t r = (cmd);                                          \
        if (r != ncclSuccess) {                                          \
            std::fprintf(stderr,                                         \
                "NCCL error %s at %s:%d\n",                              \
                ncclGetErrorString(r), __FILE__, __LINE__);              \
            std::exit(EXIT_FAILURE);                                     \
        }                                                                \
    } while (0)
#endif

inline void init_nccl_comm(ncclComm_t *nccl_comm, MPI_Comm mpi_comm)
{
    SBD_NVTX_RANGE_COLOR("init_nccl_comm", __LINE__);
    ncclUniqueId id;
    SBD_CHECK_NCCL(ncclGetUniqueId(&id));
    {
        SBD_NVTX_RANGE_COLOR("MPI_Bcast", 0);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm);
    }
    int mpi_size; MPI_Comm_size(mpi_comm, &mpi_size);
    int mpi_rank; MPI_Comm_rank(mpi_comm, &mpi_rank);
    SBD_CHECK_NCCL(ncclCommInitRank(nccl_comm, mpi_size, id, mpi_rank));
}

template <typename ElemT>
void nccl_allreduce(ElemT* buff, size_t count, ncclRedOp_t nccl_op, ncclComm_t nccl_comm, cudaStream_t stream = 0)
{
    SBD_NVTX_RANGE_COLOR("ncclAllReduce", 0);
    SBD_CHECK_NCCL(ncclAllReduce(buff, buff, count, SbdNcclDataType<ElemT>::value, nccl_op, nccl_comm, stream));
    // SBD_CHECK_CUDA(cudaStreamSynchronize(stream));
}

template <typename ElemT>
void nccl_allreduce(thrust::device_vector<ElemT> &A, ncclRedOp_t nccl_op, ncclComm_t nccl_comm, cudaStream_t stream = 0)
{
    SBD_NVTX_RANGE_COLOR("nccl_allreduce", __LINE__);
    ElemT *buff = thrust::raw_pointer_cast(A.data());
    size_t count = A.size();
    nccl_allreduce(buff, count, nccl_op, nccl_comm, stream);
}

template <typename ElemT>
void nccl_allreduce2(thrust::device_vector<ElemT>& A,
                     thrust::device_vector<ElemT>& B,
                     ncclRedOp_t nccl_op,
                     ncclComm_t nccl_comm,
                     void* ws_ptr = nullptr,
                     cudaStream_t stream = 0)
{
    SBD_NVTX_RANGE_COLOR("nccl_allreduce2", __LINE__);
    const size_t nA = A.size();
    const size_t nB = B.size();
    const size_t nTotal = nA + nB;
    if (nTotal == 0) return;

    thrust::device_vector<ElemT> ws_local;
    ElemT* packed_ptr = nullptr;
    if (ws_ptr == nullptr) {
        ws_local.resize(nTotal);
        packed_ptr = thrust::raw_pointer_cast(ws_local.data());
    } else {
        packed_ptr = static_cast<ElemT*>(ws_ptr);
    }
    ElemT* A_ptr = nA ? thrust::raw_pointer_cast(A.data()) : nullptr;
    ElemT* B_ptr = nB ? thrust::raw_pointer_cast(B.data()) : nullptr;

    // pack
    if (nA > 0) {
        SBD_CHECK_CUDA(cudaMemcpyAsync(packed_ptr, A_ptr, sizeof(ElemT) * nA,
                                       cudaMemcpyDeviceToDevice, stream));
    }
    if (nB > 0) {
        SBD_CHECK_CUDA(cudaMemcpyAsync(packed_ptr + nA, B_ptr, sizeof(ElemT) * nB,
                                       cudaMemcpyDeviceToDevice, stream));
    }

    // single allreduce on packed buffer
    nccl_allreduce(packed_ptr, nTotal, nccl_op, nccl_comm, stream);

    // unpack
    if (nA > 0) {
        SBD_CHECK_CUDA(cudaMemcpyAsync(A_ptr, packed_ptr, sizeof(ElemT) * nA,
                                       cudaMemcpyDeviceToDevice, stream));
    }
    if (nB > 0) {
        SBD_CHECK_CUDA(cudaMemcpyAsync(B_ptr, packed_ptr + nA, sizeof(ElemT) * nB,
                                       cudaMemcpyDeviceToDevice, stream));
    }
    // SBD_CHECK_CUDA(cudaStreamSynchronize(stream));
}

#endif // SBD_USE_NCCL
#endif // SBD_FRAMEWORK_NCCL_UTILITY_H
