/// Device on which a tensor's data resides.
///
/// `Meta` is a special device that does not allocate any backing memory:
/// meta tensors carry shape, dtype, and device information but no data.
/// They are useful for shape inference, dry-run model construction, and
/// inspecting parameter counts of huge models without actually allocating
/// the weights. Mirrors `torch.device("meta")`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU main memory.
    #[default]
    Cpu,
    /// CUDA GPU with the given device index.
    Cuda(usize),
    /// Meta device — shape-only, no backing storage. Operations that need
    /// data return an error; operations that only manipulate metadata
    /// (reshape, view, permute, narrow, transpose, …) work normally and
    /// produce meta tensors as output. CL-395.
    Meta,
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

    /// Returns `true` if this is the meta device (shape-only, no data).
    #[inline]
    pub fn is_meta(&self) -> bool {
        matches!(self, Device::Meta)
    }
}

impl core::fmt::Display for Device {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(id) => write!(f, "cuda:{id}"),
            Device::Meta => write!(f, "meta"),
        }
    }
}
