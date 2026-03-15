use std::collections::HashMap;
use std::path::Path;

use ferrotorch_core::FerrotorchResult;

use crate::event::ProfileEvent;

/// Summary of a single operation name across all recorded events.
#[derive(Debug, Clone)]
pub struct OpSummary {
    /// Operation name.
    pub name: String,
    /// Number of times the operation was recorded.
    pub count: usize,
    /// Cumulative time in microseconds.
    pub total_us: u64,
    /// Average time per invocation in microseconds.
    pub avg_us: u64,
    /// Maximum single-invocation time in microseconds.
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

    /// Top operations sorted by cumulative time (descending).
    pub fn top_ops(&self, n: usize) -> Vec<OpSummary> {
        let mut map: HashMap<&str, (usize, u64, u64)> = HashMap::new();
        for event in &self.events {
            let entry = map.entry(event.name.as_str()).or_insert((0, 0, 0));
            entry.0 += 1;
            entry.1 += event.duration_us;
            entry.2 = entry.2.max(event.duration_us);
        }
        let mut summaries: Vec<OpSummary> = map
            .into_iter()
            .map(|(name, (count, total_us, max_us))| {
                let avg_us = if count > 0 { total_us / count as u64 } else { 0 };
                OpSummary {
                    name: name.to_owned(),
                    count,
                    total_us,
                    avg_us,
                    max_us,
                }
            })
            .collect();
        summaries.sort_by(|a, b| b.total_us.cmp(&a.total_us));
        summaries.truncate(n);
        summaries
    }

    /// Human-readable table of the top `top_n` operations by cumulative time.
    ///
    /// Example output:
    /// ```text
    /// +---------+-------+----------+--------+--------+
    /// | Op      | Count | Total us | Avg us | Max us |
    /// +---------+-------+----------+--------+--------+
    /// | matmul  |    10 |     5000 |    500 |    800 |
    /// | relu    |    10 |     1000 |    100 |    150 |
    /// +---------+-------+----------+--------+--------+
    /// ```
    pub fn table(&self, top_n: usize) -> String {
        let ops = self.top_ops(top_n);
        if ops.is_empty() {
            return "(no events recorded)".to_owned();
        }

        // Compute column widths.
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

    /// Produce a Chrome trace JSON string suitable for `chrome://tracing`.
    ///
    /// Each event becomes an `"X"` (complete) trace event:
    /// ```json
    /// {"name":"mm","cat":"tensor_op","ph":"X","ts":1000,"dur":500,
    ///  "pid":1,"tid":1,"args":{"shapes":"[[32,784],[784,256]]"}}
    /// ```
    pub fn chrome_trace_json(&self) -> String {
        let mut buf = String::from("{\"traceEvents\":[");
        for (i, event) in self.events.iter().enumerate() {
            if i > 0 {
                buf.push(',');
            }
            let shapes_str = format_shapes(&event.input_shapes);
            // Manually construct JSON to avoid pulling in serde_json.
            buf.push_str(&format!(
                "{{\"name\":{},\"cat\":{},\"ph\":\"X\",\"ts\":{},\"dur\":{},\
                 \"pid\":1,\"tid\":{},\"args\":{{\"shapes\":{}}}}}",
                json_string(&event.name),
                json_string(&event.category),
                event.start_us,
                event.duration_us,
                event.thread_id,
                json_string(&shapes_str),
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
