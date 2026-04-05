//! Backend-specific implementations for poulpy-cpu-ref's FFT64Ref backend.
//!
//! This module provides backend-specific trait implementations required for
//! the reference backend to work correctly with poulpy-core operations.

use poulpy_hal::api::VecZnxBigBytesOf;
use poulpy_hal::layouts::Module;
use poulpy_cpu_ref::FFT64Ref;

impl VecZnxBigBytesOf for Module<FFT64Ref> {
    fn bytes_of_vec_znx_big(&self, cols: usize, size: usize) -> usize {
        self.n() * cols * size * std::mem::size_of::<i64>()
    }
}
