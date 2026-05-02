use std::collections::HashMap;
use std::path::Path;

use ferrotorch_core::FerrotorchResult;

use crate::event::{DeviceType, ProfileEvent};

/// Summary of a single operation name across all recorded events.
#[derive(Debug, Clone)]
pub struct OpSummary {
    /// Operation name.
    pub name: String,
    /// Number of times the operation was recorded.
    pub count: usize,
    /// Cumulative CPU time in microseconds (events with `DeviceType::Cpu`).
    pub cpu_total_us: u64,
    /// Average CPU time per CPU invocation in microseconds.
    pub cpu_avg_us: u64,
    /// Maximum single CPU invocation time in microseconds.
    pub cpu_max_us: u64,
    /// Number of CPU-timed invocations.
    pub cpu_count: usize,
    /// Cumulative GPU (CUDA) time in microseconds (events with `DeviceType::Cuda`).
    pub gpu_total_us: u64,
    /// Average GPU time per GPU invocation in microseconds.
    pub gpu_avg_us: u64,
    /// Maximum single GPU invocation time in microseconds.
    pub gpu_max_us: u64,
    /// Number of GPU-timed invocations.
    pub gpu_count: usize,
    /// Combined total time in microseconds (CPU + GPU).
    pub total_us: u64,
    // --- backwards-compat aliases (sum of both device types) ---
    /// Average time per invocation in microseconds (all device types).
    pub avg_us: u64,
    /// Maximum single-invocation time in microseconds (all device types).
    pub max_us: u64,
}

/// Per-operation accumulator used internally by [`ProfileReport::top_ops`].
#[derive(Default)]
struct OpAccum {
    count: usize,
    cpu_count: usize,
    cpu_total: u64,
    cpu_max: u64,
    gpu_count: usize,
    gpu_total: u64,
    gpu_max: u64,
}

/// The result of a profiling session.
///
/// Provides human-readable tables, Chrome trace JSON export, and
/// programmatic access to per-operation summaries.
pub struct ProfileReport {
    events: Vec<ProfileEvent>,
}

impl ProfileReport {
    /// Create a report from a list of events.
    pub(crate) fn new(events: Vec<ProfileEvent>) -> Self {
        Self { events }
    }

    /// All recorded events, in insertion order.
    pub fn events(&self) -> &[ProfileEvent] {
        &self.events
    }

    /// Total profiled time (sum of all event durations) in microseconds.
    pub fn total_time_us(&self) -> u64 {
        self.events.iter().map(|e| e.duration_us).sum()
    }

    /// Whether any events were recorded with CUDA timing.
    pub fn has_gpu_events(&self) -> bool {
        self.events
            .iter()
            .any(|e| e.device_type == DeviceType::Cuda)
    }

    /// Sum of estimated FLOPS across all events that had a FLOPS
    /// estimate. Events with `flops == None` (op not recognized,
    /// shapes not recorded) are skipped. CL-333.
    pub fn total_flops(&self) -> u64 {
        self.events.iter().filter_map(|e| e.flops).sum()
    }

    /// Estimated total FLOPS divided by the wall-clock CPU time
    /// (microseconds) of the run, expressed as FLOPS per second
    /// (i.e. multiply by 1e6 from MFLOPS, which is what dividing by
    /// microseconds gives you). Returns 0 when there are no events
    /// or no time elapsed. CL-333.
    pub fn flops_per_second(&self) -> f64 {
        let total = self.total_flops() as f64;
        let elapsed_us = self.total_time_us() as f64;
        if elapsed_us > 0.0 {
            total * 1_000_000.0 / elapsed_us
        } else {
            0.0
        }
    }

    /// Group memory events by their `MemoryCategory`, returning
    /// `(category, net_bytes)` pairs. Allocations are positive, frees
    /// are negative; the result is the running net allocation per
    /// category. CL-333.
    pub fn memory_by_category(&self) -> Vec<(crate::event::MemoryCategory, i64)> {
        let mut totals: HashMap<crate::event::MemoryCategory, i64> = HashMap::new();
        for event in &self.events {
            if let (Some(bytes), Some(cat)) = (event.memory_bytes, event.memory_category) {
                *totals.entry(cat).or_insert(0) += bytes;
            }
        }
        let mut out: Vec<(crate::event::MemoryCategory, i64)> = totals.into_iter().collect();
        // Sort by absolute byte size descending so largest category first.
        out.sort_by_key(|b| std::cmp::Reverse(b.1.abs()));
        out
    }

    /// Whether any events have a non-empty stack trace. CL-333.
    pub fn has_stack_traces(&self) -> bool {
        self.events.iter().any(|e| e.stack_trace.is_some())
    }

    /// Top operations sorted by cumulative time (descending).
    pub fn top_ops(&self, n: usize) -> Vec<OpSummary> {
        // Per-op accumulator.
        let mut map: HashMap<&str, OpAccum> = HashMap::new();
        for event in &self.events {
            let entry = map.entry(event.name.as_str()).or_default();
            entry.count += 1;
            match event.device_type {
                DeviceType::Cpu => {
                    entry.cpu_count += 1;
                    entry.cpu_total += event.duration_us;
                    entry.cpu_max = entry.cpu_max.max(event.duration_us);
                }
                DeviceType::Cuda => {
                    entry.gpu_count += 1;
                    entry.gpu_total += event.duration_us;
                    entry.gpu_max = entry.gpu_max.max(event.duration_us);
                }
            }
        }
        let mut summaries: Vec<OpSummary> = map
            .into_iter()
            .map(|(name, acc)| {
                let OpAccum {
                    count,
                    cpu_count,
                    cpu_total,
                    cpu_max,
                    gpu_count,
                    gpu_total,
                    gpu_max,
                } = acc;
                let cpu_avg = if cpu_count > 0 {
                    cpu_total / cpu_count as u64
                } else {
                    0
                };
                let gpu_avg = if gpu_count > 0 {
                    gpu_total / gpu_count as u64
                } else {
                    0
                };
                let total = cpu_total + gpu_total;
                let avg = if count > 0 { total / count as u64 } else { 0 };
                let max = cpu_max.max(gpu_max);
                OpSummary {
                    name: name.to_owned(),
                    count,
                    cpu_total_us: cpu_total,
                    cpu_avg_us: cpu_avg,
                    cpu_max_us: cpu_max,
                    cpu_count,
                    gpu_total_us: gpu_total,
                    gpu_avg_us: gpu_avg,
                    gpu_max_us: gpu_max,
                    gpu_count,
                    total_us: total,
                    avg_us: avg,
                    max_us: max,
                }
            })
            .collect();
        summaries.sort_by_key(|b| std::cmp::Reverse(b.total_us));
        summaries.truncate(n);
        summaries
    }

    /// Human-readable table of the top `top_n` operations by cumulative time.
    ///
    /// When GPU events are present, CPU and GPU times are shown as separate
    /// columns. When all events are CPU-only, the table omits the GPU columns
    /// for backwards compatibility.
    ///
    /// Example output (CPU-only):
    /// ```text
    /// +---------+-------+----------+--------+--------+
    /// | Op      | Count | Total us | Avg us | Max us |
    /// +---------+-------+----------+--------+--------+
    /// | matmul  |    10 |     5000 |    500 |    800 |
    /// | relu    |    10 |     1000 |    100 |    150 |
    /// +---------+-------+----------+--------+--------+
    /// ```
    ///
    /// Example output (with GPU events):
    /// ```text
    /// +---------+-------+--------+--------------+-----------+---------+--------------+-----------+
    /// | Op      | Count | Device | CPU Total us | CPU Avg   | CPU Max | GPU Total us | GPU Avg   |
    /// +---------+-------+--------+--------------+-----------+---------+--------------+-----------+
    /// | matmul  |    10 | CUDA   |            0 |         0 |       0 |         5000 |       500 |
    /// | relu    |    10 | CPU    |         1000 |       100 |     150 |            0 |         0 |
    /// +---------+-------+--------+--------------+-----------+---------+--------------+-----------+
    /// ```
    pub fn table(&self, top_n: usize) -> String {
        let ops = self.top_ops(top_n);
        if ops.is_empty() {
            return "(no events recorded)".to_owned();
        }

        let has_gpu = self.has_gpu_events();

        if has_gpu {
            self.table_with_gpu(&ops)
        } else {
            self.table_cpu_only(&ops)
        }
    }

    /// Render a CPU-only table (legacy format, backwards compatible).
    fn table_cpu_only(&self, ops: &[OpSummary]) -> String {
        let hdr = ["Op", "Count", "Total us", "Avg us", "Max us"];
        let rows: Vec<[String; 5]> = ops
            .iter()
            .map(|o| {
                [
                    o.name.clone(),
                    o.count.to_string(),
                    o.total_us.to_string(),
                    o.avg_us.to_string(),
                    o.max_us.to_string(),
                ]
            })
            .collect();

        let mut widths = [0usize; 5];
        for (i, h) in hdr.iter().enumerate() {
            widths[i] = h.len();
        }
        for row in &rows {
            for (i, cell) in row.iter().enumerate() {
                widths[i] = widths[i].max(cell.len());
            }
        }

        let sep = format!(
            "+-{}-+-{}-+-{}-+-{}-+-{}-+",
            "-".repeat(widths[0]),
            "-".repeat(widths[1]),
            "-".repeat(widths[2]),
            "-".repeat(widths[3]),
            "-".repeat(widths[4]),
        );

        let header = format!(
            "| {:<w0$} | {:>w1$} | {:>w2$} | {:>w3$} | {:>w4$} |",
            hdr[0],
            hdr[1],
            hdr[2],
            hdr[3],
            hdr[4],
            w0 = widths[0],
            w1 = widths[1],
            w2 = widths[2],
            w3 = widths[3],
            w4 = widths[4],
        );

        let mut out = String::new();
        out.push_str(&sep);
        out.push('\n');
        out.push_str(&header);
        out.push('\n');
        out.push_str(&sep);
        out.push('\n');
        for row in &rows {
            let line = format!(
                "| {:<w0$} | {:>w1$} | {:>w2$} | {:>w3$} | {:>w4$} |",
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                w0 = widths[0],
                w1 = widths[1],
                w2 = widths[2],
                w3 = widths[3],
                w4 = widths[4],
            );
            out.push_str(&line);
            out.push('\n');
        }
        out.push_str(&sep);
        out
    }

    /// Render a table with separate CPU and GPU columns.
    fn table_with_gpu(&self, ops: &[OpSummary]) -> String {
        let hdr = [
            "Op",
            "Count",
            "Device",
            "CPU Total us",
            "CPU Avg us",
            "CPU Max us",
            "GPU Total us",
            "GPU Avg us",
            "GPU Max us",
        ];
        let rows: Vec<[String; 9]> = ops
            .iter()
            .map(|o| {
                // Show the dominant device type.
                let device = if o.gpu_count > 0 && o.cpu_count > 0 {
                    "CPU+CUDA".to_owned()
                } else if o.gpu_count > 0 {
                    "CUDA".to_owned()
                } else {
                    "CPU".to_owned()
                };
                [
                    o.name.clone(),
                    o.count.to_string(),
                    device,
                    o.cpu_total_us.to_string(),
                    o.cpu_avg_us.to_string(),
                    o.cpu_max_us.to_string(),
                    o.gpu_total_us.to_string(),
                    o.gpu_avg_us.to_string(),
                    o.gpu_max_us.to_string(),
                ]
            })
            .collect();

        let mut widths = [0usize; 9];
        for (i, h) in hdr.iter().enumerate() {
            widths[i] = h.len();
        }
        for row in &rows {
            for (i, cell) in row.iter().enumerate() {
                widths[i] = widths[i].max(cell.len());
            }
        }

        let sep = {
            let mut s = String::from("+");
            for w in &widths {
                s.push('-');
                for _ in 0..*w {
                    s.push('-');
                }
                s.push_str("-+");
            }
            s
        };

        let header = {
            let mut s = String::from("|");
            // First column (Op) is left-aligned, rest are right-aligned.
            s.push_str(&format!(" {:<w$} |", hdr[0], w = widths[0]));
            for i in 1..9 {
                s.push_str(&format!(" {:>w$} |", hdr[i], w = widths[i]));
            }
            s
        };

        let mut out = String::new();
        out.push_str(&sep);
        out.push('\n');
        out.push_str(&header);
        out.push('\n');
        out.push_str(&sep);
        out.push('\n');
        for row in &rows {
            let mut line = String::from("|");
            line.push_str(&format!(" {:<w$} |", row[0], w = widths[0]));
            for i in 1..9 {
                line.push_str(&format!(" {:>w$} |", row[i], w = widths[i]));
            }
            out.push_str(&line);
            out.push('\n');
        }
        out.push_str(&sep);
        out
    }

    /// Produce a Chrome trace JSON string suitable for `chrome://tracing`.
    ///
    /// Each event becomes an `"X"` (complete) trace event. GPU events include
    /// a `"device":"CUDA"` arg.
    pub fn chrome_trace_json(&self) -> String {
        let mut buf = String::from("{\"traceEvents\":[");
        for (i, event) in self.events.iter().enumerate() {
            if i > 0 {
                buf.push(',');
            }
            let shapes_str = format_shapes(&event.input_shapes);
            let device_str = event.device_type.to_string();
            // Manually construct JSON to avoid pulling in serde_json.
            buf.push_str(&format!(
                "{{\"name\":{},\"cat\":{},\"ph\":\"X\",\"ts\":{},\"dur\":{},\
                 \"pid\":1,\"tid\":{},\"args\":{{\"shapes\":{},\"device\":{}}}}}",
                json_string(&event.name),
                json_string(&event.category),
                event.start_us,
                event.duration_us,
                event.thread_id,
                json_string(&shapes_str),
                json_string(&device_str),
            ));
        }
        buf.push_str("]}");
        buf
    }

    /// Write Chrome trace JSON to a file.
    pub fn save_chrome_trace(&self, path: impl AsRef<Path>) -> FerrotorchResult<()> {
        let json = self.chrome_trace_json();
        std::fs::write(path.as_ref(), json).map_err(|e| {
            ferrotorch_core::FerrotorchError::InvalidArgument {
                message: format!("failed to write chrome trace: {e}"),
            }
        })
    }

    /// Export this profile report to a TensorBoard-readable directory
    /// layout under `logdir`. CL-381.
    ///
    /// TensorBoard's PyTorch Profiler plugin (available via
    /// `tensorboard --logdir <path>` with the `torch_tb_profiler`
    /// package installed) reads Chrome-format trace files from a
    /// specific directory structure:
    ///
    /// ```text
    /// {logdir}/plugins/profile/{run_id}/{hostname}.pt.trace.json
    /// ```
    ///
    /// This method creates that structure under `logdir` (creating
    /// intermediate directories as needed) and writes the Chrome trace
    /// JSON to the expected filename. The `run_id` is typically a
    /// wall-clock timestamp; it defaults to `"run0"` if not provided,
    /// giving a stable default. The `hostname` argument is recorded in
    /// the filename so multi-node runs don't collide; it defaults to
    /// the system hostname via `gethostname`-equivalent environment
    /// lookup, falling back to `"localhost"`.
    ///
    /// Returns the absolute path of the written trace file.
    ///
    /// # Arguments
    ///
    /// - `logdir` — TensorBoard log directory root. Must exist or be
    ///   createable (this method creates it if missing).
    /// - `run_id` — label for this profiling run, becomes the
    ///   intermediate directory name under `plugins/profile/`. Pass
    ///   `None` to use `"run0"`.
    /// - `hostname` — hostname label baked into the filename. Pass
    ///   `None` to auto-detect.
    pub fn save_tensorboard_trace(
        &self,
        logdir: impl AsRef<Path>,
        run_id: Option<&str>,
        hostname: Option<&str>,
    ) -> FerrotorchResult<std::path::PathBuf> {
        use ferrotorch_core::FerrotorchError;

        let logdir = logdir.as_ref();
        let run_id = run_id.unwrap_or("run0");
        let default_hostname;
        let hostname = match hostname {
            Some(h) => h,
            None => {
                default_hostname = detect_hostname();
                &default_hostname
            }
        };

        // Build the target directory: {logdir}/plugins/profile/{run_id}/
        let target_dir = logdir.join("plugins").join("profile").join(run_id);
        std::fs::create_dir_all(&target_dir).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("save_tensorboard_trace: failed to create {target_dir:?}: {e}"),
        })?;

        let filename = format!("{hostname}.pt.trace.json");
        let target_file = target_dir.join(filename);
        let json = self.chrome_trace_json();
        std::fs::write(&target_file, json).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("save_tensorboard_trace: failed to write {target_file:?}: {e}"),
        })?;
        Ok(target_file)
    }
}

/// Best-effort hostname lookup for the TensorBoard trace filename.
/// Checks common environment variables; falls back to `"localhost"`
/// when none are set. Avoids the `hostname` crate dependency for a
/// feature that doesn't need exact hostnames.
fn detect_hostname() -> String {
    for var in &["HOSTNAME", "COMPUTERNAME", "HOST"] {
        if let Ok(h) = std::env::var(var) {
            if !h.is_empty() {
                return h;
            }
        }
    }
    "localhost".to_string()
}

/// Format shapes like `[[32,784],[784,256]]`.
fn format_shapes(shapes: &[Vec<usize>]) -> String {
    let mut buf = String::from("[");
    for (i, shape) in shapes.iter().enumerate() {
        if i > 0 {
            buf.push(',');
        }
        buf.push('[');
        for (j, &dim) in shape.iter().enumerate() {
            if j > 0 {
                buf.push(',');
            }
            buf.push_str(&dim.to_string());
        }
        buf.push(']');
    }
    buf.push(']');
    buf
}

/// Minimalist JSON string escaping (no serde dependency).
fn json_string(s: &str) -> String {
    let mut buf = String::with_capacity(s.len() + 2);
    buf.push('"');
    for c in s.chars() {
        match c {
            '"' => buf.push_str("\\\""),
            '\\' => buf.push_str("\\\\"),
            '\n' => buf.push_str("\\n"),
            '\r' => buf.push_str("\\r"),
            '\t' => buf.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                buf.push_str(&format!("\\u{:04x}", c as u32));
            }
            _ => buf.push(c),
        }
    }
    buf.push('"');
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_shapes_empty() {
        assert_eq!(format_shapes(&[]), "[]");
    }

    #[test]
    fn test_format_shapes_single() {
        assert_eq!(format_shapes(&[vec![3, 4]]), "[[3,4]]");
    }

    #[test]
    fn test_format_shapes_multiple() {
        assert_eq!(
            format_shapes(&[vec![32, 784], vec![784, 256]]),
            "[[32,784],[784,256]]"
        );
    }

    #[test]
    fn test_json_string_escaping() {
        assert_eq!(json_string("hello"), "\"hello\"");
        assert_eq!(json_string("a\"b"), "\"a\\\"b\"");
        assert_eq!(json_string("a\\b"), "\"a\\\\b\"");
        assert_eq!(json_string("a\nb"), "\"a\\nb\"");
    }

    /// Both detect_hostname tests mutate the same set of env vars,
    /// so they must not run concurrently. Cargo runs tests in
    /// parallel by default, so we serialize them through this
    /// process-wide lock. The previous version of these tests
    /// raced under `cargo test --features cuda` because the test
    /// matrix was larger than the no-feature build.
    static HOSTNAME_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_detect_hostname_fallback() {
        let _g = HOSTNAME_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // Defensive check: when the common env vars are unset we fall
        // back to "localhost". Setting them all to empty strings
        // should also trigger the fallback since we reject empty
        // values.
        let original = std::env::var("HOSTNAME").ok();
        let original_computer = std::env::var("COMPUTERNAME").ok();
        let original_host = std::env::var("HOST").ok();
        // SAFETY: serialized via HOSTNAME_TEST_LOCK above; no other
        // thread touches these env vars while the lock is held.
        unsafe {
            std::env::remove_var("HOSTNAME");
            std::env::remove_var("COMPUTERNAME");
            std::env::remove_var("HOST");
        }
        let h = detect_hostname();
        assert_eq!(h, "localhost");
        // Restore to avoid affecting other tests.
        if let Some(v) = original {
            unsafe { std::env::set_var("HOSTNAME", v) };
        }
        if let Some(v) = original_computer {
            unsafe { std::env::set_var("COMPUTERNAME", v) };
        }
        if let Some(v) = original_host {
            unsafe { std::env::set_var("HOST", v) };
        }
    }

    #[test]
    fn test_detect_hostname_uses_env_var() {
        let _g = HOSTNAME_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let original = std::env::var("HOSTNAME").ok();
        unsafe {
            std::env::set_var("HOSTNAME", "my-test-host");
        }
        assert_eq!(detect_hostname(), "my-test-host");
        // Restore.
        match original {
            Some(v) => unsafe { std::env::set_var("HOSTNAME", v) },
            None => unsafe { std::env::remove_var("HOSTNAME") },
        }
    }
}
