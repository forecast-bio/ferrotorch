/// Device on which a tensor's data resides.
///
/// Defined in Phase 1 with only `Cpu` functional. `Cuda` is present
/// from day one so the type is baked into every API before GPU work begins.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU main memory.
    #[default]
    Cpu,
    /// CUDA GPU with the given device index.
    Cuda(usize),
}

impl Device {
    /// Returns `true` if this is a CPU device.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Returns `true` if this is a CUDA device.
    #[inline]
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }
}

impl core::fmt::Display for Device {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(id) => write!(f, "cuda:{id}"),
        }
    }
}

