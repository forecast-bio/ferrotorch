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
        self.events.iter().any(|e| e.device_type == DeviceType::Cuda)
    }

    /// Top operations sorted by cumulative time (descending).
    pub fn top_ops(&self, n: usize) -> Vec<OpSummary> {
        // Per-op accumulator: (count, cpu_count, cpu_total, cpu_max, gpu_count, gpu_total, gpu_max)
        let mut map: HashMap<&str, (usize, usize, u64, u64, usize, u64, u64)> = HashMap::new();
        for event in &self.events {
            let entry = map
                .entry(event.name.as_str())
                .or_insert((0, 0, 0, 0, 0, 0, 0));
            entry.0 += 1; // total count
            match event.device_type {
                DeviceType::Cpu => {
                    entry.1 += 1; // cpu_count
                    entry.2 += event.duration_us; // cpu_total
                    entry.3 = entry.3.max(event.duration_us); // cpu_max
                }
                DeviceType::Cuda => {
                    entry.4 += 1; // gpu_count
                    entry.5 += event.duration_us; // gpu_total
                    entry.6 = entry.6.max(event.duration_us); // gpu_max
                }
            }
        }
        let mut summaries: Vec<OpSummary> = map
            .into_iter()
            .map(
                |(name, (count, cpu_count, cpu_total, cpu_max, gpu_count, gpu_total, gpu_max))| {
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
                    let avg = if count > 0 {
                        total / count as u64
                    } else {
                        0
                    };
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
                },
            )
            .collect();
        summaries.sort_by(|a, b| b.total_us.cmp(&a.total_us));
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
}
