// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "axpy/src/args.h"
#include "kmeans/src/args.h"
#include "atax/src/args.h"
#include "correlation/src/args.h"
#include "covariance/src/args.h"

typedef struct {
    volatile uint32_t local_job_addr;
} usr_data_t;

//////////
// AXPY //
//////////

typedef struct {
    uint32_t id;
    uint8_t offload_id;
    axpy_args_t args;
} axpy_job_t;

/////////////
// K-Means //
/////////////

typedef struct {
    uint32_t id;
    uint8_t offload_id;
    kmeans_args_t args;
} kmeans_job_t;

//////////
// GEMM //
//////////

typedef struct {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint64_t a_ptr;
    uint64_t b_ptr;
    uint64_t c_ptr;
} offload_gemm_args_t;

typedef struct {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint64_t a_l3_ptr;
    uint64_t b_l3_ptr;
    uint64_t c_l3_ptr;
    double* a;
    double* b;
    double* c;
} gemm_local_args_t;

typedef struct {
    uint32_t id;
    uint8_t offload_id;
    offload_gemm_args_t args;
} gemm_job_t;

typedef struct {
    uint32_t id;
    uint8_t offload_id;
    gemm_local_args_t args;
} gemm_local_job_t;

/////////////////
// Monte Carlo //
/////////////////

typedef struct {
    uint32_t n_samples;
    uint64_t result_ptr;
} mc_args_t;

typedef struct {
    uint32_t id;
    uint8_t offload_id;
    mc_args_t args;
} mc_job_t;

//////////
// ATAX //
//////////

typedef struct {
    uint32_t id;
    uint8_t offload_id;
    atax_args_t args;
} atax_job_t;

/////////////////
// Correlation //
/////////////////

typedef struct {
    uint32_t id;
    uint8_t offload_id;
    correlation_args_t args;
} correlation_job_t;

////////////////
// Covariance //
////////////////

typedef struct {
    uint32_t id;
    uint8_t offload_id;
    covariance_args_t args;
} covariance_job_t;

/////////////
// Generic //
/////////////

typedef struct {
    uint64_t job_ptr;
} user_data_t;

typedef union {
    axpy_args_t axpy;
    offload_gemm_args_t gemm;
    mc_args_t mc;
    kmeans_args_t kmeans;
    atax_args_t atax;
    correlation_args_t correlation;
    covariance_args_t covariance;
} job_args_t;

typedef struct {
    uint32_t id;
    uint8_t offload_id;
    job_args_t args;
} job_t;

#define N_JOB_TYPES 7
typedef enum {
    J_AXPY = 0,
    J_GEMM = 1,
    J_MONTECARLO = 2,
    J_KMEANS = 3,
    J_ATAX = 4,
    J_CORRELATION = 5,
    J_COVARIANCE = 6
} job_id_t;

static inline uint32_t job_args_size(job_id_t job_id) {
    switch (job_id) {
    case J_AXPY:
        return sizeof(axpy_args_t);
    case J_GEMM:
        return sizeof(offload_gemm_args_t);
    case J_MONTECARLO:
        return sizeof(mc_args_t);
    case J_KMEANS:
        return sizeof(kmeans_args_t);
    case J_ATAX:
        return sizeof(atax_args_t);
    case J_CORRELATION:
        return sizeof(correlation_args_t);
    case J_COVARIANCE:
        return sizeof(covariance_args_t);
    default:
        return 0;
    }
}
