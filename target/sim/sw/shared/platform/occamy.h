// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "occamy_cfg.h"
#include "occamy_memory_map.h"

#ifdef N_CLUSTERS_TO_USE
#define N_CLUSTERS N_CLUSTERS_TO_USE
#else
#define N_CLUSTERS (N_QUADS * N_CLUSTERS_PER_QUAD)
#endif
#define N_SNITCHES (N_CLUSTERS * N_CORES_PER_CLUSTER)
#define N_HARTS (N_SNITCHES + 1)
