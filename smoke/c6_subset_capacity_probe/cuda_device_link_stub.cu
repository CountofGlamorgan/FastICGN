/*
 * Copyright (c) 2026 Chonglang Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

// Empty kernel that marks this target as CUDA code, so CMake device-links the
// separable-compiled FastICGN static library into the probe executable.
__global__ void deviceLinkStubKernel()
{
}
