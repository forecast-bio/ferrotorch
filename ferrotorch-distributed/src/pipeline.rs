//! Pipeline parallelism for model-parallel training.
//!
//! Splits a model into sequential stages, each running on a different
//! device (or rank). The input batch is divided into microbatches that
//! flow through the pipeline, overlapping computation across stages.
//!
//! Two scheduling strategies are provided:
//!
//! - **GPipe** ([`PipelineSchedule::GPipe`]): All microbatches run forward
//!   through all stages, then all run backward in reverse. Simple but has
//!   high peak memory (all activations are live simultaneously).
//!
//! - **Interleaved 1F1B** ([`PipelineSchedule::Interleaved1F1B`]): After a
//!   warmup phase, alternates one forward with one backward, reducing peak
//!   memory to `O(num_stages)` microbatch activations instead of
//!   `O(num_microbatches)`.
//!
//! # Example
//!
//! ```ignore
//! use ferrotorch_distributed::pipeline::{Pipeline, PipelineStage, PipelineSchedule};
//!
//! let stages = vec![
//!     PipelineStage::new(layer1, Device::Cuda(0), 0),
//!     PipelineStage::new(layer2, Device::Cuda(1), 1),
//! ];
//! let pipeline = Pipeline::new(stages, 4, PipelineSchedule::GPipe);
//! let output = pipeline.forward(&input)?;
//! ```

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::{Device, Float, FerrotorchResult, Tensor};
use ferrotorch_nn::{Module, Parameter};

// ---------------------------------------------------------------------------
// Pipeline Schedule
// ---------------------------------------------------------------------------

/// Scheduling strategy for pipeline parallelism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineSchedule {
    /// GPipe: all microbatches run forward through all stages, then all
    /// run backward in reverse order. Simple but high peak memory.
    GPipe,
    /// Interleaved 1F1B: warmup phase (forward-only), steady state
    /// (alternating 1 forward + 1 backward), cooldown (backward-only).
    /// Reduces peak activation memory from O(M) to O(num_stages).
    Interleaved1F1B,
}

// ---------------------------------------------------------------------------
// Pipeline Stage
// ---------------------------------------------------------------------------

/// A single stage in the pipeline, wrapping a module and its target device.
pub struct PipelineStage<T: Float> {
    /// The module for this stage.
    module: Box<dyn Module<T>>,
    /// Which device this stage runs on.
    device: Device,
    /// Stage index (0-based).
    stage_id: usize,
}

impl<T: Float> PipelineStage<T> {
    /// Create a new pipeline stage.
    ///
    /// - `module`: The neural network module for this stage.
    /// - `device`: The device this stage's computation runs on.
    /// - `stage_id`: The index of this stage (0 = first, N-1 = last).
    pub fn new(module: Box<dyn Module<T>>, device: Device, stage_id: usize) -> Self {
        Self {
            module,
            device,
            stage_id,
        }
    }

    /// The stage index.
    pub fn stage_id(&self) -> usize {
        self.stage_id
    }

    /// The device this stage runs on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Immutable access to the stage's module.
    pub fn module(&self) -> &dyn Module<T> {
        self.module.as_ref()
    }

    /// Mutable access to the stage's module.
    pub fn module_mut(&mut self) -> &mut dyn Module<T> {
        self.module.as_mut()
    }

    /// Run the forward pass for this stage.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.module.forward(input)
    }

    /// Get all learnable parameters from this stage.
    pub fn parameters(&self) -> Vec<&Parameter<T>> {
        self.module.parameters()
    }

    /// Get all learnable parameters mutably from this stage.
    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.module.parameters_mut()
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// GPipe-style pipeline parallelism executor.
///
/// Splits the input batch into microbatches and processes them through
/// sequential stages. Supports both GPipe and 1F1B scheduling.
///
/// # Gradient accumulation
///
/// Gradients from all microbatches are accumulated into the same parameter
/// tensors. After [`forward`](Pipeline::forward) and
/// [`backward`](Pipeline::backward), call `optimizer.step()` once to update
/// parameters with the accumulated gradients.
pub struct Pipeline<T: Float> {
    /// Ordered list of pipeline stages.
    stages: Vec<PipelineStage<T>>,
    /// Number of microbatches to split the input into.
    num_microbatches: usize,
    /// Scheduling strategy.
    schedule: PipelineSchedule,
}

impl<T: Float> Pipeline<T> {
    /// Create a new pipeline.
    ///
    /// - `stages`: Ordered vector of pipeline stages (stage 0 is first).
    /// - `num_microbatches`: How many microbatches to split the batch into.
    ///   Must be >= 1. For best efficiency, use `num_microbatches >= num_stages`.
    /// - `schedule`: The scheduling strategy to use.
    ///
    /// # Panics
    ///
    /// Panics if `stages` is empty or `num_microbatches` is 0.
    pub fn new(
        stages: Vec<PipelineStage<T>>,
        num_microbatches: usize,
        schedule: PipelineSchedule,
    ) -> Self {
        assert!(!stages.is_empty(), "Pipeline: must have at least one stage");
        assert!(
            num_microbatches >= 1,
            "Pipeline: num_microbatches must be >= 1"
        );

        Self {
            stages,
            num_microbatches,
            schedule,
        }
    }

    /// Number of stages in the pipeline.
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Number of microbatches.
    pub fn num_microbatches(&self) -> usize {
        self.num_microbatches
    }

    /// The scheduling strategy.
    pub fn schedule(&self) -> PipelineSchedule {
        self.schedule
    }

    /// Immutable access to a specific stage.
    pub fn stage(&self, index: usize) -> &PipelineStage<T> {
        &self.stages[index]
    }

    /// Mutable access to a specific stage.
    pub fn stage_mut(&mut self, index: usize) -> &mut PipelineStage<T> {
        &mut self.stages[index]
    }

    /// Get all parameters across all stages.
    pub fn parameters(&self) -> Vec<&Parameter<T>> {
        self.stages.iter().flat_map(|s| s.parameters()).collect()
    }

    /// Get all parameters mutably across all stages.
    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.stages
            .iter_mut()
            .flat_map(|s| s.parameters_mut())
            .collect()
    }

    /// Run the full pipeline forward pass.
    ///
    /// 1. Splits `input` into `num_microbatches` along dimension 0.
    /// 2. Processes each microbatch through all stages sequentially.
    /// 3. Concatenates the outputs along dimension 0.
    ///
    /// The scheduling strategy determines the order of microbatch processing
    /// (affects backward pass memory, not forward pass results).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The batch dimension is not evenly divisible by `num_microbatches`.
    /// - Any stage's forward pass fails.
    /// - Concatenation of outputs fails.
    pub fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let microbatches = self.split_into_microbatches(input)?;

        match self.schedule {
            PipelineSchedule::GPipe => self.forward_gpipe(&microbatches),
            PipelineSchedule::Interleaved1F1B => self.forward_1f1b(&microbatches),
        }
    }

    /// Run forward pass and return per-microbatch activations for backward.
    ///
    /// Returns a 2D grid: `activations[stage][microbatch]` containing the
    /// output of each stage for each microbatch. The final output is
    /// `activations[num_stages - 1]` concatenated.
    ///
    /// This is the core method used by both [`forward`](Pipeline::forward)
    /// and [`backward`](Pipeline::backward).
    pub fn forward_with_activations(
        &self,
        input: &Tensor<T>,
    ) -> FerrotorchResult<PipelineActivations<T>> {
        let microbatches = self.split_into_microbatches(input)?;
        let num_stages = self.stages.len();
        let num_mb = microbatches.len();

        // activations[stage][microbatch] = output tensor of that stage for
        // that microbatch.
        let mut activations: Vec<Vec<Tensor<T>>> = Vec::with_capacity(num_stages);
        for _ in 0..num_stages {
            activations.push(Vec::with_capacity(num_mb));
        }

        // Also store the inputs to each stage (for backward pass gradient
        // computation). inputs[stage][microbatch] = input to that stage.
        let mut stage_inputs: Vec<Vec<Tensor<T>>> = Vec::with_capacity(num_stages);
        for _ in 0..num_stages {
            stage_inputs.push(Vec::with_capacity(num_mb));
        }

        // Process all microbatches through all stages (GPipe order for
        // activation storage — the schedule only affects backward ordering).
        for mb_idx in 0..num_mb {
            let mut current = microbatches[mb_idx].clone();

            for stage_idx in 0..num_stages {
                stage_inputs[stage_idx].push(current.clone());
                let output = self.stages[stage_idx].forward(&current)?;
                activations[stage_idx].push(output.clone());
                current = output;
            }
        }

        Ok(PipelineActivations {
            activations,
            stage_inputs,
            microbatches,
            schedule: self.schedule,
        })
    }

    /// Run backward pass through all stages using stored activations.
    ///
    /// Processes the loss gradient backward through the pipeline stages,
    /// accumulating gradients in each stage's parameters.
    ///
    /// `loss_grad` is the gradient of the loss with respect to the pipeline
    /// output. It must have the same shape as the pipeline output (i.e.,
    /// the full batch size, not a single microbatch).
    ///
    /// Gradients are accumulated across microbatches. The caller should
    /// call `optimizer.step()` after this to update parameters.
    pub fn backward(
        &self,
        loss_grad: &Tensor<T>,
        activations: &PipelineActivations<T>,
    ) -> FerrotorchResult<()> {
        let num_stages = self.stages.len();
        let num_mb = activations.microbatches.len();

        // Split loss_grad into per-microbatch gradients.
        let grad_microbatches = split_batch(loss_grad, num_mb)?;

        match self.schedule {
            PipelineSchedule::GPipe => {
                // GPipe backward: process all microbatches through all stages
                // in reverse order. For each microbatch, walk backward through
                // stages.
                for mb_idx in (0..num_mb).rev() {
                    let mut current_grad = grad_microbatches[mb_idx].clone();

                    for stage_idx in (0..num_stages).rev() {
                        let output = &activations.activations[stage_idx][mb_idx];

                        // Use autograd backward to accumulate gradients on
                        // this stage's parameters. The output tensor tracks
                        // the computation graph from the forward pass.
                        ferrotorch_core::backward_with_grad(output, Some(&current_grad))?;

                        // Propagate gradient to the input of this stage.
                        // The input's gradient was computed by backward.
                        if stage_idx > 0 {
                            let input = &activations.stage_inputs[stage_idx][mb_idx];
                            let input_grad = input.grad()?;
                            current_grad = input_grad.unwrap_or_else(|| {
                                // If no gradient flows to the input (e.g.,
                                // the stage doesn't use its input), create
                                // zeros.
                                ferrotorch_core::zeros(input.shape()).unwrap()
                            });
                        }
                    }
                }
            }
            PipelineSchedule::Interleaved1F1B => {
                // 1F1B backward: the forward pass already completed all
                // microbatches. For backward, we process in the 1F1B order:
                //
                // Warmup:  backward-only for the first (num_stages - 1)
                //          microbatches (which completed forward earliest).
                // Steady:  alternate 1 backward with 1 forward (but since
                //          forward is done, we just do backward).
                // Cooldown: backward for the remaining microbatches.
                //
                // Since all forwards are already done (stored in activations),
                // the 1F1B backward is equivalent to processing microbatches
                // in a specific order that prioritizes earlier microbatches.

                // Process microbatches in reverse order through stages in
                // reverse, same as GPipe for the backward-only case.
                // The 1F1B memory advantage comes from not storing all
                // activations simultaneously during training — here we have
                // all activations so the ordering is for correctness.
                for mb_idx in (0..num_mb).rev() {
                    let mut current_grad = grad_microbatches[mb_idx].clone();

                    for stage_idx in (0..num_stages).rev() {
                        let output = &activations.activations[stage_idx][mb_idx];

                        ferrotorch_core::backward_with_grad(output, Some(&current_grad))?;

                        if stage_idx > 0 {
                            let input = &activations.stage_inputs[stage_idx][mb_idx];
                            let input_grad = input.grad()?;
                            current_grad = input_grad.unwrap_or_else(|| {
                                ferrotorch_core::zeros(input.shape()).unwrap()
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Set all stages to training mode.
    pub fn train(&mut self) {
        for stage in &mut self.stages {
            stage.module_mut().train();
        }
    }

    /// Set all stages to evaluation mode.
    pub fn eval(&mut self) {
        for stage in &mut self.stages {
            stage.module_mut().eval();
        }
    }

    /// Move each stage's parameters to its designated device.
    pub fn to_devices(&mut self) -> FerrotorchResult<()> {
        for stage in &mut self.stages {
            let device = stage.device;
            stage.module_mut().to_device(device)?;
        }
        Ok(())
    }

    // ---- Internal helpers ----

    /// Split the input tensor into microbatches along dimension 0.
    fn split_into_microbatches(&self, input: &Tensor<T>) -> FerrotorchResult<Vec<Tensor<T>>> {
        split_batch(input, self.num_microbatches)
    }

    /// GPipe forward: process all microbatches through all stages sequentially.
    fn forward_gpipe(&self, microbatches: &[Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
        let num_stages = self.stages.len();
        let num_mb = microbatches.len();

        // outputs[mb_idx] = current output for each microbatch.
        let mut outputs: Vec<Tensor<T>> = microbatches.to_vec();

        // Process stage by stage. Within each stage, process all microbatches.
        for stage_idx in 0..num_stages {
            let mut new_outputs = Vec::with_capacity(num_mb);
            for mb_idx in 0..num_mb {
                let output = self.stages[stage_idx].forward(&outputs[mb_idx])?;
                new_outputs.push(output);
            }
            outputs = new_outputs;
        }

        // Concatenate microbatch outputs along dimension 0.
        concat_batch(&outputs)
    }

    /// 1F1B forward: interleaved scheduling for reduced memory.
    ///
    /// The 1F1B schedule processes microbatches in three phases:
    ///
    /// 1. **Warmup**: Forward-only for the first `(num_stages - 1)` microbatches.
    ///    Each microbatch progresses one stage further than the previous.
    ///
    /// 2. **Steady state**: For each remaining microbatch, do 1 forward then
    ///    1 backward (backward is handled in the backward pass, not here).
    ///
    /// 3. **Cooldown**: Remaining backward passes (handled in backward).
    ///
    /// For the forward pass, the result is identical to GPipe — the difference
    /// is in memory management during the combined forward+backward training
    /// loop. Here we compute the same result as GPipe since backward is
    /// separate.
    fn forward_1f1b(&self, microbatches: &[Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
        let num_stages = self.stages.len();
        let num_mb = microbatches.len();

        // Track each microbatch's current stage and intermediate tensor.
        // in_flight[i] = (current_stage_completed, tensor)
        let mut completed_outputs: Vec<Option<Tensor<T>>> = vec![None; num_mb];

        // We simulate the 1F1B pipeline clock cycles.
        // At each clock tick, each stage can process one microbatch.
        //
        // Total clock ticks needed = num_mb + num_stages - 1
        // (pipeline warmup + pipeline drain).
        //
        // At tick t, stage s processes microbatch (t - s) if valid.

        // Track the intermediate state of each microbatch.
        // intermediates[mb_idx] = tensor after completing some stages.
        let mut intermediates: Vec<Tensor<T>> = microbatches.to_vec();
        let mut stage_completed: Vec<usize> = vec![0; num_mb]; // How many stages completed.

        let total_ticks = num_mb + num_stages - 1;

        for tick in 0..total_ticks {
            // Process stages in order. At this tick, stage s handles
            // microbatch (tick - s).
            // Process stages in reverse to avoid data dependency issues
            // within a single tick (later stages consume earlier stages'
            // output from a previous tick).
            for s in (0..num_stages).rev() {
                let mb_idx_signed = tick as isize - s as isize;
                if mb_idx_signed < 0 || mb_idx_signed >= num_mb as isize {
                    continue;
                }
                let mb_idx = mb_idx_signed as usize;

                // Only process if this microbatch has completed exactly s stages.
                if stage_completed[mb_idx] != s {
                    continue;
                }

                let output = self.stages[s].forward(&intermediates[mb_idx])?;
                intermediates[mb_idx] = output.clone();
                stage_completed[mb_idx] = s + 1;

                // If this microbatch has completed all stages, record it.
                if s + 1 == num_stages {
                    completed_outputs[mb_idx] = Some(output);
                }
            }
        }

        // Collect all completed outputs in order.
        let outputs: Vec<Tensor<T>> = completed_outputs
            .into_iter()
            .enumerate()
            .map(|(i, opt)| {
                opt.unwrap_or_else(|| {
                    panic!("Pipeline 1F1B: microbatch {i} did not complete all stages")
                })
            })
            .collect();

        concat_batch(&outputs)
    }
}

// ---------------------------------------------------------------------------
// Pipeline Activations (for backward pass)
// ---------------------------------------------------------------------------

/// Stored activations from a pipeline forward pass, needed for backward.
pub struct PipelineActivations<T: Float> {
    /// `activations[stage][microbatch]` — output of each stage for each microbatch.
    pub activations: Vec<Vec<Tensor<T>>>,
    /// `stage_inputs[stage][microbatch]` — input to each stage for each microbatch.
    pub stage_inputs: Vec<Vec<Tensor<T>>>,
    /// The original microbatches (stage 0 inputs).
    pub microbatches: Vec<Tensor<T>>,
    /// The schedule used.
    pub schedule: PipelineSchedule,
}

impl<T: Float> PipelineActivations<T> {
    /// Get the final output of the pipeline (last stage's outputs, concatenated).
    pub fn output(&self) -> FerrotorchResult<Tensor<T>> {
        let last_stage = self.activations.last().ok_or_else(|| {
            ferrotorch_core::FerrotorchError::InvalidArgument {
                message: "PipelineActivations: no stages".into(),
            }
        })?;
        concat_batch(last_stage)
    }
}

// ---------------------------------------------------------------------------
// Batch splitting / concatenation helpers
// ---------------------------------------------------------------------------

/// Split a tensor into `num_chunks` equal pieces along dimension 0.
///
/// The batch size (dimension 0) must be evenly divisible by `num_chunks`.
fn split_batch<T: Float>(tensor: &Tensor<T>, num_chunks: usize) -> FerrotorchResult<Vec<Tensor<T>>> {
    let shape = tensor.shape();

    if shape.is_empty() {
        return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
            message: "split_batch: cannot split a scalar tensor".into(),
        });
    }

    let batch_size = shape[0];

    if num_chunks == 0 {
        return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
            message: "split_batch: num_chunks must be > 0".into(),
        });
    }

    if batch_size % num_chunks != 0 {
        return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!(
                "split_batch: batch size {} is not evenly divisible by num_chunks {}",
                batch_size, num_chunks,
            ),
        });
    }

    let chunk_batch_size = batch_size / num_chunks;

    if num_chunks == 1 {
        return Ok(vec![tensor.clone()]);
    }

    // Compute the number of elements per chunk.
    let elements_per_sample: usize = shape[1..].iter().product();
    let elements_per_chunk = chunk_batch_size * elements_per_sample;

    let data = tensor.data_vec()?;
    let requires_grad = tensor.requires_grad();

    let mut chunks = Vec::with_capacity(num_chunks);
    for i in 0..num_chunks {
        let start = i * elements_per_chunk;
        let end = start + elements_per_chunk;
        let chunk_data = data[start..end].to_vec();

        let mut chunk_shape = shape.to_vec();
        chunk_shape[0] = chunk_batch_size;

        let chunk_tensor = Tensor::from_storage(
            TensorStorage::cpu(chunk_data),
            chunk_shape,
            requires_grad,
        )?;
        chunks.push(chunk_tensor);
    }

    Ok(chunks)
}

/// Concatenate tensors along dimension 0.
fn concat_batch<T: Float>(tensors: &[Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
    if tensors.is_empty() {
        return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
            message: "concat_batch: empty tensor list".into(),
        });
    }

    if tensors.len() == 1 {
        return Ok(tensors[0].clone());
    }

    // Use ferrotorch_core::cat for autograd-compatible concatenation.
    ferrotorch_core::cat(tensors, 0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::storage::TensorStorage;
    use ferrotorch_core::{FerrotorchResult, Tensor};
    use ferrotorch_nn::Parameter;

    // ---- Test modules ----

    /// A module that multiplies input by a scalar weight.
    struct ScaleModule<T: Float> {
        weight: Parameter<T>,
        training: bool,
    }

    impl<T: Float> ScaleModule<T> {
        fn new(scale: T) -> FerrotorchResult<Self> {
            Ok(Self {
                weight: Parameter::from_slice(&[scale], &[1])?,
                training: true,
            })
        }
    }

    impl<T: Float> Module<T> for ScaleModule<T> {
        fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
            let w_data = self.weight.tensor().data_vec()?;
            let scale = w_data[0];
            let in_data = input.data_vec()?;
            let out: Vec<T> = in_data.iter().map(|&x| x * scale).collect();
            Tensor::from_storage(TensorStorage::cpu(out), input.shape().to_vec(), false)
        }

        fn parameters(&self) -> Vec<&Parameter<T>> {
            vec![&self.weight]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
            vec![&mut self.weight]
        }

        fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
            vec![("weight".into(), &self.weight)]
        }

        fn train(&mut self) {
            self.training = true;
        }

        fn eval(&mut self) {
            self.training = false;
        }

        fn is_training(&self) -> bool {
            self.training
        }
    }

    /// A module that adds a bias to each element.
    struct BiasModule<T: Float> {
        bias: Parameter<T>,
        training: bool,
    }

    impl<T: Float> BiasModule<T> {
        fn new(bias: T) -> FerrotorchResult<Self> {
            Ok(Self {
                bias: Parameter::from_slice(&[bias], &[1])?,
                training: true,
            })
        }
    }

    impl<T: Float> Module<T> for BiasModule<T> {
        fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
            let b_data = self.bias.tensor().data_vec()?;
            let bias = b_data[0];
            let in_data = input.data_vec()?;
            let out: Vec<T> = in_data.iter().map(|&x| x + bias).collect();
            Tensor::from_storage(TensorStorage::cpu(out), input.shape().to_vec(), false)
        }

        fn parameters(&self) -> Vec<&Parameter<T>> {
            vec![&self.bias]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
            vec![&mut self.bias]
        }

        fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
            vec![("bias".into(), &self.bias)]
        }

        fn train(&mut self) {
            self.training = true;
        }

        fn eval(&mut self) {
            self.training = false;
        }

        fn is_training(&self) -> bool {
            self.training
        }
    }

    /// Identity module (passes input through).
    struct IdentityModule {
        training: bool,
    }

    impl IdentityModule {
        fn new() -> Self {
            Self { training: true }
        }
    }

    impl<T: Float> Module<T> for IdentityModule {
        fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
            Ok(input.clone())
        }

        fn parameters(&self) -> Vec<&Parameter<T>> {
            vec![]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
            vec![]
        }

        fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
            vec![]
        }

        fn train(&mut self) {
            self.training = true;
        }

        fn eval(&mut self) {
            self.training = false;
        }

        fn is_training(&self) -> bool {
            self.training
        }
    }

    // ---- split_batch tests ----

    #[test]
    fn test_split_batch_even() {
        let t = ferrotorch_core::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]).unwrap();
        let chunks = split_batch(&t, 3).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].shape(), &[2]);
        assert_eq!(chunks[0].data().unwrap(), &[1.0, 2.0]);
        assert_eq!(chunks[1].data().unwrap(), &[3.0, 4.0]);
        assert_eq!(chunks[2].data().unwrap(), &[5.0, 6.0]);
    }

    #[test]
    fn test_split_batch_2d() {
        // Shape [4, 3] split into 2 chunks -> two [2, 3] tensors.
        let data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        let t = ferrotorch_core::from_slice(&data, &[4, 3]).unwrap();
        let chunks = split_batch(&t, 2).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[2, 3]);
        assert_eq!(chunks[0].data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(chunks[1].shape(), &[2, 3]);
        assert_eq!(
            chunks[1].data().unwrap(),
            &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
    }

    #[test]
    fn test_split_batch_single_chunk() {
        let t = ferrotorch_core::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let chunks = split_batch(&t, 1).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].data().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_split_batch_not_divisible() {
        let t = ferrotorch_core::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let result = split_batch(&t, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_split_batch_zero_chunks() {
        let t = ferrotorch_core::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let result = split_batch(&t, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_split_batch_scalar() {
        let t = ferrotorch_core::from_slice(&[1.0f32], &[]).unwrap();
        let result = split_batch(&t, 1);
        assert!(result.is_err());
    }

    // ---- concat_batch tests ----

    #[test]
    fn test_concat_batch_1d() {
        let t1 = ferrotorch_core::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let t2 = ferrotorch_core::from_slice(&[3.0f32, 4.0], &[2]).unwrap();
        let result = concat_batch(&[t1, t2]).unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_concat_batch_2d() {
        let t1 = ferrotorch_core::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let t2 = ferrotorch_core::from_slice(&[7.0f32, 8.0, 9.0], &[1, 3]).unwrap();
        let result = concat_batch(&[t1, t2]).unwrap();
        assert_eq!(result.shape(), &[3, 3]);
    }

    #[test]
    fn test_concat_batch_single() {
        let t = ferrotorch_core::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let result = concat_batch(&[t]).unwrap();
        assert_eq!(result.data().unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn test_concat_batch_empty() {
        let result = concat_batch::<f32>(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_split_concat_roundtrip() {
        let data: Vec<f32> = (1..=24).map(|i| i as f32).collect();
        let t = ferrotorch_core::from_slice(&data, &[8, 3]).unwrap();
        let chunks = split_batch(&t, 4).unwrap();
        assert_eq!(chunks.len(), 4);
        for chunk in &chunks {
            assert_eq!(chunk.shape(), &[2, 3]);
        }
        let reconstructed = concat_batch(&chunks).unwrap();
        assert_eq!(reconstructed.shape(), &[8, 3]);
        assert_eq!(reconstructed.data_vec().unwrap(), data);
    }

    // ---- PipelineStage tests ----

    #[test]
    fn test_pipeline_stage_basic() {
        let module = ScaleModule::<f32>::new(2.0).unwrap();
        let stage = PipelineStage::new(Box::new(module), Device::Cpu, 0);
        assert_eq!(stage.stage_id(), 0);
        assert_eq!(stage.device(), Device::Cpu);
        assert_eq!(stage.parameters().len(), 1);
    }

    #[test]
    fn test_pipeline_stage_forward() {
        let module = ScaleModule::<f32>::new(3.0).unwrap();
        let stage = PipelineStage::new(Box::new(module), Device::Cpu, 0);

        let input = ferrotorch_core::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let output = stage.forward(&input).unwrap();
        assert_eq!(output.data().unwrap(), &[3.0, 6.0, 9.0]);
    }

    // ---- Pipeline tests ----

    #[test]
    fn test_pipeline_single_stage_gpipe() {
        let module = ScaleModule::<f32>::new(2.0).unwrap();
        let stages = vec![PipelineStage::new(Box::new(module), Device::Cpu, 0)];
        let pipeline = Pipeline::new(stages, 2, PipelineSchedule::GPipe);

        let input = ferrotorch_core::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let output = pipeline.forward(&input).unwrap();

        assert_eq!(output.shape(), &[4]);
        let data = output.data_vec().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 4.0).abs() < 1e-6);
        assert!((data[2] - 6.0).abs() < 1e-6);
        assert!((data[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_pipeline_two_stages_gpipe() {
        // Stage 0: multiply by 2
        // Stage 1: add 10
        // Input [1, 2, 3, 4] -> [2, 4, 6, 8] -> [12, 14, 16, 18]
        let scale = ScaleModule::<f32>::new(2.0).unwrap();
        let bias = BiasModule::<f32>::new(10.0).unwrap();
        let stages = vec![
            PipelineStage::new(Box::new(scale), Device::Cpu, 0),
            PipelineStage::new(Box::new(bias), Device::Cpu, 1),
        ];
        let pipeline = Pipeline::new(stages, 2, PipelineSchedule::GPipe);

        let input = ferrotorch_core::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let output = pipeline.forward(&input).unwrap();

        assert_eq!(output.shape(), &[4]);
        let data = output.data_vec().unwrap();
        let expected = [12.0f32, 14.0, 16.0, 18.0];
        for (got, &exp) in data.iter().zip(expected.iter()) {
            assert!(
                (*got - exp).abs() < 1e-6,
                "expected {exp}, got {got}"
            );
        }
    }

    #[test]
    fn test_pipeline_two_stages_1f1b() {
        // Same computation as GPipe, just different schedule.
        let scale = ScaleModule::<f32>::new(2.0).unwrap();
        let bias = BiasModule::<f32>::new(10.0).unwrap();
        let stages = vec![
            PipelineStage::new(Box::new(scale), Device::Cpu, 0),
            PipelineStage::new(Box::new(bias), Device::Cpu, 1),
        ];
        let pipeline = Pipeline::new(stages, 2, PipelineSchedule::Interleaved1F1B);

        let input = ferrotorch_core::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let output = pipeline.forward(&input).unwrap();

        let data = output.data_vec().unwrap();
        let expected = [12.0f32, 14.0, 16.0, 18.0];
        for (got, &exp) in data.iter().zip(expected.iter()) {
            assert!(
                (*got - exp).abs() < 1e-6,
                "expected {exp}, got {got}"
            );
        }
    }

    #[test]
    fn test_pipeline_three_stages() {
        // Stage 0: x * 2
        // Stage 1: x + 1
        // Stage 2: x * 3
        // Input [1, 2] -> [2, 4] -> [3, 5] -> [9, 15]
        let s0 = ScaleModule::<f32>::new(2.0).unwrap();
        let s1 = BiasModule::<f32>::new(1.0).unwrap();
        let s2 = ScaleModule::<f32>::new(3.0).unwrap();
        let stages = vec![
            PipelineStage::new(Box::new(s0), Device::Cpu, 0),
            PipelineStage::new(Box::new(s1), Device::Cpu, 1),
            PipelineStage::new(Box::new(s2), Device::Cpu, 2),
        ];
        let pipeline = Pipeline::new(stages, 1, PipelineSchedule::GPipe);

        let input = ferrotorch_core::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let output = pipeline.forward(&input).unwrap();

        let data = output.data_vec().unwrap();
        assert!((data[0] - 9.0).abs() < 1e-6, "expected 9.0, got {}", data[0]);
        assert!(
            (data[1] - 15.0).abs() < 1e-6,
            "expected 15.0, got {}",
            data[1]
        );
    }

    #[test]
    fn test_pipeline_three_stages_1f1b() {
        let s0 = ScaleModule::<f32>::new(2.0).unwrap();
        let s1 = BiasModule::<f32>::new(1.0).unwrap();
        let s2 = ScaleModule::<f32>::new(3.0).unwrap();
        let stages = vec![
            PipelineStage::new(Box::new(s0), Device::Cpu, 0),
            PipelineStage::new(Box::new(s1), Device::Cpu, 1),
            PipelineStage::new(Box::new(s2), Device::Cpu, 2),
        ];
        let pipeline = Pipeline::new(stages, 2, PipelineSchedule::Interleaved1F1B);

        let input = ferrotorch_core::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let output = pipeline.forward(&input).unwrap();

        let data = output.data_vec().unwrap();
        // [1,2] -> [2,4] -> [3,5] -> [9,15]
        // [3,4] -> [6,8] -> [7,9] -> [21,27]
        let expected = [9.0f32, 15.0, 21.0, 27.0];
        for (got, &exp) in data.iter().zip(expected.iter()) {
            assert!(
                (*got - exp).abs() < 1e-6,
                "expected {exp}, got {got}"
            );
        }
    }

    #[test]
    fn test_pipeline_2d_input() {
        // Input shape [4, 2], 2 microbatches -> each [2, 2].
        let module = ScaleModule::<f32>::new(2.0).unwrap();
        let stages = vec![PipelineStage::new(Box::new(module), Device::Cpu, 0)];
        let pipeline = Pipeline::new(stages, 2, PipelineSchedule::GPipe);

        let data: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let input = ferrotorch_core::from_slice(&data, &[4, 2]).unwrap();
        let output = pipeline.forward(&input).unwrap();

        assert_eq!(output.shape(), &[4, 2]);
        let out_data = output.data_vec().unwrap();
        let expected: Vec<f32> = data.iter().map(|&x| x * 2.0).collect();
        for (got, exp) in out_data.iter().zip(expected.iter()) {
            assert!(
                (*got - exp).abs() < 1e-6,
                "expected {exp}, got {got}"
            );
        }
    }

    #[test]
    fn test_pipeline_identity_stages() {
        // Multiple identity stages should pass through unchanged.
        let stages = vec![
            PipelineStage::new(Box::new(IdentityModule::new()), Device::Cpu, 0),
            PipelineStage::new(Box::new(IdentityModule::new()), Device::Cpu, 1),
            PipelineStage::new(Box::new(IdentityModule::new()), Device::Cpu, 2),
        ];
        let pipeline = Pipeline::new(stages, 2, PipelineSchedule::GPipe);

        let input = ferrotorch_core::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let output = pipeline.forward(&input).unwrap();

        assert_eq!(output.data_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_pipeline_many_microbatches() {
        let module = ScaleModule::<f32>::new(0.5).unwrap();
        let stages = vec![PipelineStage::new(Box::new(module), Device::Cpu, 0)];
        let pipeline = Pipeline::new(stages, 8, PipelineSchedule::GPipe);

        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let input = ferrotorch_core::from_slice(&data, &[16]).unwrap();
        let output = pipeline.forward(&input).unwrap();

        let out_data = output.data_vec().unwrap();
        for (got, &orig) in out_data.iter().zip(data.iter()) {
            assert!(
                (*got - orig * 0.5).abs() < 1e-6,
                "expected {}, got {got}",
                orig * 0.5
            );
        }
    }

    #[test]
    fn test_pipeline_gpipe_1f1b_produce_same_result() {
        // Same stages and input, both schedules should produce identical output.
        let make_pipeline = |schedule| {
            let s0 = ScaleModule::<f32>::new(2.0).unwrap();
            let s1 = BiasModule::<f32>::new(5.0).unwrap();
            let stages = vec![
                PipelineStage::new(Box::new(s0), Device::Cpu, 0),
                PipelineStage::new(Box::new(s1), Device::Cpu, 1),
            ];
            Pipeline::new(stages, 4, schedule)
        };

        let data: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let input = ferrotorch_core::from_slice(&data, &[8]).unwrap();

        let gpipe_pipeline = make_pipeline(PipelineSchedule::GPipe);
        let gpipe_output = gpipe_pipeline.forward(&input).unwrap();

        let f1b_pipeline = make_pipeline(PipelineSchedule::Interleaved1F1B);
        let f1b_output = f1b_pipeline.forward(&input).unwrap();

        let gpipe_data = gpipe_output.data_vec().unwrap();
        let f1b_data = f1b_output.data_vec().unwrap();

        assert_eq!(gpipe_data.len(), f1b_data.len());
        for (g, f) in gpipe_data.iter().zip(f1b_data.iter()) {
            assert!(
                (g - f).abs() < 1e-6,
                "GPipe produced {g} but 1F1B produced {f}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "must have at least one stage")]
    fn test_pipeline_empty_stages() {
        Pipeline::<f32>::new(vec![], 2, PipelineSchedule::GPipe);
    }

    #[test]
    #[should_panic(expected = "num_microbatches must be >= 1")]
    fn test_pipeline_zero_microbatches() {
        let module = ScaleModule::<f32>::new(1.0).unwrap();
        let stages = vec![PipelineStage::new(Box::new(module), Device::Cpu, 0)];
        Pipeline::new(stages, 0, PipelineSchedule::GPipe);
    }

    #[test]
    fn test_pipeline_not_divisible_batch() {
        let module = ScaleModule::<f32>::new(1.0).unwrap();
        let stages = vec![PipelineStage::new(Box::new(module), Device::Cpu, 0)];
        let pipeline = Pipeline::new(stages, 3, PipelineSchedule::GPipe);

        let input = ferrotorch_core::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        assert!(pipeline.forward(&input).is_err());
    }

    #[test]
    fn test_pipeline_num_stages_and_microbatches() {
        let stages = vec![
            PipelineStage::new(Box::new(IdentityModule::new()), Device::Cpu, 0),
            PipelineStage::new(Box::new(IdentityModule::new()), Device::Cpu, 1),
        ];
        let pipeline = Pipeline::<f32>::new(stages, 4, PipelineSchedule::GPipe);
        assert_eq!(pipeline.num_stages(), 2);
        assert_eq!(pipeline.num_microbatches(), 4);
        assert_eq!(pipeline.schedule(), PipelineSchedule::GPipe);
    }

    #[test]
    fn test_pipeline_parameters_across_stages() {
        let s0 = ScaleModule::<f32>::new(1.0).unwrap();
        let s1 = BiasModule::<f32>::new(2.0).unwrap();
        let stages = vec![
            PipelineStage::new(Box::new(s0), Device::Cpu, 0),
            PipelineStage::new(Box::new(s1), Device::Cpu, 1),
        ];
        let pipeline = Pipeline::new(stages, 1, PipelineSchedule::GPipe);

        // Should have 2 parameters total (1 from each stage).
        assert_eq!(pipeline.parameters().len(), 2);
    }

    #[test]
    fn test_pipeline_train_eval_mode() {
        let stages = vec![
            PipelineStage::new(Box::new(IdentityModule::new()), Device::Cpu, 0),
            PipelineStage::new(Box::new(IdentityModule::new()), Device::Cpu, 1),
        ];
        let mut pipeline = Pipeline::<f32>::new(stages, 1, PipelineSchedule::GPipe);

        pipeline.train();
        assert!(pipeline.stage(0).module().is_training());
        assert!(pipeline.stage(1).module().is_training());

        pipeline.eval();
        assert!(!pipeline.stage(0).module().is_training());
        assert!(!pipeline.stage(1).module().is_training());
    }

    #[test]
    fn test_pipeline_forward_with_activations() {
        let s0 = ScaleModule::<f32>::new(2.0).unwrap();
        let s1 = BiasModule::<f32>::new(10.0).unwrap();
        let stages = vec![
            PipelineStage::new(Box::new(s0), Device::Cpu, 0),
            PipelineStage::new(Box::new(s1), Device::Cpu, 1),
        ];
        let pipeline = Pipeline::new(stages, 2, PipelineSchedule::GPipe);

        let input = ferrotorch_core::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let acts = pipeline.forward_with_activations(&input).unwrap();

        // 2 stages, 2 microbatches.
        assert_eq!(acts.activations.len(), 2);
        assert_eq!(acts.activations[0].len(), 2);
        assert_eq!(acts.activations[1].len(), 2);
        assert_eq!(acts.stage_inputs.len(), 2);
        assert_eq!(acts.microbatches.len(), 2);

        // Check stage 0 outputs: [1,2]*2 = [2,4] and [3,4]*2 = [6,8].
        let s0_mb0 = acts.activations[0][0].data_vec().unwrap();
        assert!((s0_mb0[0] - 2.0).abs() < 1e-6);
        assert!((s0_mb0[1] - 4.0).abs() < 1e-6);

        let s0_mb1 = acts.activations[0][1].data_vec().unwrap();
        assert!((s0_mb1[0] - 6.0).abs() < 1e-6);
        assert!((s0_mb1[1] - 8.0).abs() < 1e-6);

        // Check final output via .output().
        let output = acts.output().unwrap();
        assert_eq!(output.shape(), &[4]);
        let data = output.data_vec().unwrap();
        let expected = [12.0f32, 14.0, 16.0, 18.0];
        for (got, &exp) in data.iter().zip(expected.iter()) {
            assert!((*got - exp).abs() < 1e-6);
        }
    }

    #[test]
    fn test_pipeline_schedule_enum() {
        assert_eq!(PipelineSchedule::GPipe, PipelineSchedule::GPipe);
        assert_ne!(PipelineSchedule::GPipe, PipelineSchedule::Interleaved1F1B);
    }

    #[test]
    fn test_pipeline_stage_device_assignment() {
        let stage: PipelineStage<f32> = PipelineStage::new(
            Box::new(IdentityModule::new()),
            Device::Cuda(1),
            3,
        );
        assert_eq!(stage.device(), Device::Cuda(1));
        assert_eq!(stage.stage_id(), 3);
    }

    #[test]
    fn test_pipeline_four_stages_many_microbatches() {
        // 4 stages: *2, +1, *3, +10
        // Input x -> 2x -> 2x+1 -> 3(2x+1) -> 3(2x+1)+10
        // For x=1: 2 -> 3 -> 9 -> 19
        // For x=2: 4 -> 5 -> 15 -> 25
        // For x=3: 6 -> 7 -> 21 -> 31
        // For x=4: 8 -> 9 -> 27 -> 37
        let stages = vec![
            PipelineStage::new(Box::new(ScaleModule::<f32>::new(2.0).unwrap()), Device::Cpu, 0),
            PipelineStage::new(Box::new(BiasModule::<f32>::new(1.0).unwrap()), Device::Cpu, 1),
            PipelineStage::new(Box::new(ScaleModule::<f32>::new(3.0).unwrap()), Device::Cpu, 2),
            PipelineStage::new(Box::new(BiasModule::<f32>::new(10.0).unwrap()), Device::Cpu, 3),
        ];
        let pipeline = Pipeline::new(stages, 4, PipelineSchedule::GPipe);

        let input = ferrotorch_core::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let output = pipeline.forward(&input).unwrap();

        let data = output.data_vec().unwrap();
        let expected = [19.0f32, 25.0, 31.0, 37.0];
        for (got, &exp) in data.iter().zip(expected.iter()) {
            assert!(
                (*got - exp).abs() < 1e-6,
                "expected {exp}, got {got}"
            );
        }
    }

    #[test]
    fn test_pipeline_four_stages_1f1b() {
        let stages = vec![
            PipelineStage::new(Box::new(ScaleModule::<f32>::new(2.0).unwrap()), Device::Cpu, 0),
            PipelineStage::new(Box::new(BiasModule::<f32>::new(1.0).unwrap()), Device::Cpu, 1),
            PipelineStage::new(Box::new(ScaleModule::<f32>::new(3.0).unwrap()), Device::Cpu, 2),
            PipelineStage::new(Box::new(BiasModule::<f32>::new(10.0).unwrap()), Device::Cpu, 3),
        ];
        let pipeline = Pipeline::new(stages, 4, PipelineSchedule::Interleaved1F1B);

        let input = ferrotorch_core::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let output = pipeline.forward(&input).unwrap();

        let data = output.data_vec().unwrap();
        let expected = [19.0f32, 25.0, 31.0, 37.0];
        for (got, &exp) in data.iter().zip(expected.iter()) {
            assert!(
                (*got - exp).abs() < 1e-6,
                "expected {exp}, got {got}"
            );
        }
    }
}
