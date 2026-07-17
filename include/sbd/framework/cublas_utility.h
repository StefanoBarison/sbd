/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SBD_FRAMEWORK_CUBLAS_UTILITY_H
#define SBD_FRAMEWORK_CUBLAS_UTILITY_H

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <thrust/device_vector.h>

namespace sbd {

inline const char* CublasStatusToString(cublasStatus_t status)
{
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
#ifdef CUBLAS_STATUS_NOT_SUPPORTED
    case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#ifdef CUBLAS_STATUS_LICENSE_ERROR
    case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
    default:                             return "CUBLAS_STATUS_UNKNOWN";
    }
}

#ifndef SBD_CHECK_CUBLAS
#define SBD_CHECK_CUBLAS(cmd)                                             \
    do {                                                                  \
        cublasStatus_t s = (cmd);                                         \
        if (s != CUBLAS_STATUS_SUCCESS) {                                 \
            std::fprintf(stderr,                                          \
                "cuBLAS error %s at %s:%d\n",                             \
                ::sbd::CublasStatusToString(s), __FILE__, __LINE__);      \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)
#endif

class CublasContext
{
private:
    cublasHandle_t handle_;

public:
    CublasContext() : handle_(nullptr)
    {
        SBD_CHECK_CUBLAS(cublasCreate(&handle_));
        SBD_CHECK_CUBLAS(
            cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST));
    }

    ~CublasContext()
    {
        if (handle_ != nullptr) {
            (void)cublasDestroy(handle_);
        }
    }

    CublasContext(const CublasContext&) = delete;
    CublasContext& operator=(const CublasContext&) = delete;

    CublasContext(CublasContext&& other) noexcept : handle_(other.handle_)
    {
        other.handle_ = nullptr;
    }

    CublasContext& operator=(CublasContext&& other) noexcept
    {
        if (this != &other) {
            if (handle_ != nullptr) {
                (void)cublasDestroy(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    cublasHandle_t get() const
    {
        return handle_;
    }

    void set_pointer_mode_host() const
    {
        SBD_CHECK_CUBLAS(
            cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST));
    }

    void set_pointer_mode_device() const
    {
        SBD_CHECK_CUBLAS(
            cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_DEVICE));
    }

    void set_stream(cudaStream_t stream) const
    {
        SBD_CHECK_CUBLAS(cublasSetStream(handle_, stream));
    }
};

inline CublasContext& GetCublasContext()
{
    thread_local CublasContext ctx;
    return ctx;
}

inline cublasHandle_t GetCublasHandle()
{
    return GetCublasContext().get();
}

template <typename ElemT>
struct CublasDot;

template <>
struct CublasDot<float>
{
    static cublasStatus_t call(cublasHandle_t handle,
                               int n,
                               const float* x,
                               int incx,
                               const float* y,
                               int incy,
                               float* result)
    {
        return cublasSdot(handle, n, x, incx, y, incy, result);
    }
};

template <>
struct CublasDot<double>
{
    static cublasStatus_t call(cublasHandle_t handle,
                               int n,
                               const double* x,
                               int incx,
                               const double* y,
                               int incy,
                               double* result)
    {
        return cublasDdot(handle, n, x, incx, y, incy, result);
    }
};

template <typename T>
inline const T* raw_ptr(const thrust::device_vector<T>& v)
{
    return thrust::raw_pointer_cast(v.data());
}

template <typename T>
inline T* raw_ptr(thrust::device_vector<T>& v)
{
    return thrust::raw_pointer_cast(v.data());
}

template <typename ElemT>
inline ElemT Dot(const thrust::device_vector<ElemT>& x,
                 const thrust::device_vector<ElemT>& y,
                 cudaStream_t stream = 0)
{
    if (x.size() != y.size()) {
        throw std::runtime_error("sbd::Dot: size mismatch");
    }
    if (x.empty()) {
        return ElemT{};
    }
    if (x.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("sbd::Dot: vector size exceeds INT_MAX");
    }

    auto& ctx = GetCublasContext();
    ctx.set_stream(stream);
    ctx.set_pointer_mode_host();
    ElemT result{};
    SBD_CHECK_CUBLAS(
        CublasDot<ElemT>::call(
            ctx.get(),
            static_cast<int>(x.size()),
            raw_ptr(x), 1,
            raw_ptr(y), 1,
            &result));

    return result;
}

template <typename ElemT>
inline void Dot(const thrust::device_vector<ElemT>& x,
                const thrust::device_vector<ElemT>& y,
                ElemT* result_dev_ptr,
                cudaStream_t stream = 0)
{
    if (x.size() != y.size()) {
        throw std::runtime_error("sbd::Dot: size mismatch");
    }
    if (x.empty()) {
        return;
    }
    if (x.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("sbd::Dot: vector size exceeds INT_MAX");
    }

    auto& ctx = GetCublasContext();
    ctx.set_stream(stream);
    ctx.set_pointer_mode_device();
    SBD_CHECK_CUBLAS(
        CublasDot<ElemT>::call(
            ctx.get(),
            static_cast<int>(x.size()),
            raw_ptr(x), 1,
            raw_ptr(y), 1,
            result_dev_ptr));
}

} // namespace sbd

#endif // SBD_FRAMEWORK_CUBLAS_UTILITY_H
