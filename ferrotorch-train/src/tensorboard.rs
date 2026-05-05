//! TensorBoard event file writing.
//!
//! This module writes TFEvents binary files that TensorBoard can read. It
//! implements a minimal protobuf encoder for the `Event` / `Summary` messages
//! and the CRC32C framing that TFEvents requires.
//!
//! # File format
//!
//! Each record in a TFEvents file is encoded as:
//!
//! ```text
//! [uint64 length][uint32 masked_crc32c(length)][bytes data][uint32 masked_crc32c(data)]
//! ```
//!
//! The `data` payload is a protobuf-encoded `Event` message.
//!
//! # Example
//!
//! ```ignore
//! use ferrotorch_train::TensorBoardWriter;
//!
//! let mut writer = TensorBoardWriter::new("runs/experiment_1")?;
//! writer.add_scalar("loss/train", 0.5, 0)?;
//! writer.add_scalar("loss/train", 0.3, 1)?;
//! writer.flush()?;
//! ```

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use ferrotorch_core::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};

use crate::callback::Callback;
use crate::history::EpochResult;

// ===========================================================================
// CRC32C (Castagnoli)
// ===========================================================================

/// CRC32C lookup table, generated from the Castagnoli polynomial 0x82F6_3B78.
const CRC32C_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0x82F6_3B78;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
};

/// Compute the CRC32C checksum of `data`.
fn crc32c(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32C_TABLE[index];
    }
    crc ^ 0xFFFF_FFFF
}

/// TFEvents masked CRC32C: rotate and add a constant so that the stored
/// checksum can be verified without knowing the original length encoding.
fn masked_crc32c(data: &[u8]) -> u32 {
    let crc = crc32c(data);
    crc.rotate_right(15).wrapping_add(0xa282_ead8)
}

// ===========================================================================
// Minimal protobuf writer (same pattern as ferrotorch-serialize/onnx_export)
// ===========================================================================

/// Lightweight protobuf wire-format writer for TFEvents messages.
///
/// Only the wire types needed for TensorBoard events are implemented:
/// - varint (wire type 0)
/// - 64-bit fixed (wire type 1)
/// - length-delimited (wire type 2)
/// - 32-bit fixed (wire type 5)
struct ProtobufWriter {
    buf: Vec<u8>,
}

// `ProtobufWriter` is the lightweight wire-format encoder for TFEvents
// messages. The full helper set (`write_int64`, `write_string`,
// `write_message`, `write_float`, `write_double`, `bytes`) is intentionally
// kept available for forward compatibility — adding new event/summary
// fields in this module should not need to touch the encoder. Only a
// subset is exercised by the current `encode_*` functions, hence the
// `dead_code` allow.
#[allow(dead_code)]
impl ProtobufWriter {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    fn bytes(&self) -> &[u8] {
        &self.buf
    }

    // -- low-level primitives -----------------------------------------------

    fn write_varint(&mut self, mut value: u64) {
        loop {
            let byte = (value & 0x7F) as u8;
            value >>= 7;
            if value == 0 {
                self.buf.push(byte);
                return;
            }
            self.buf.push(byte | 0x80);
        }
    }

    fn write_tag(&mut self, field_number: u32, wire_type: u32) {
        self.write_varint(((field_number as u64) << 3) | wire_type as u64);
    }

    // -- typed field writers -------------------------------------------------

    /// Write an `int64` field (wire type 0 -- varint).
    fn write_int64(&mut self, field_number: u32, value: i64) {
        self.write_tag(field_number, 0);
        self.write_varint(value as u64);
    }

    /// Write a `string` field (wire type 2 -- length-delimited).
    fn write_string(&mut self, field_number: u32, s: &str) {
        self.write_tag(field_number, 2);
        self.write_varint(s.len() as u64);
        self.buf.extend_from_slice(s.as_bytes());
    }

    /// Write an embedded message field (wire type 2 -- length-delimited).
    fn write_message(&mut self, field_number: u32, message: &[u8]) {
        self.write_tag(field_number, 2);
        self.write_varint(message.len() as u64);
        self.buf.extend_from_slice(message);
    }

    /// Write a `float` field (wire type 5 -- 32-bit fixed).
    fn write_float(&mut self, field_number: u32, value: f32) {
        self.write_tag(field_number, 5);
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Write a `double` field (wire type 1 -- 64-bit fixed).
    fn write_double(&mut self, field_number: u32, value: f64) {
        self.write_tag(field_number, 1);
        self.buf.extend_from_slice(&value.to_le_bytes());
    }
}

// ===========================================================================
// Protobuf message builders
// ===========================================================================

/// Encode a `Summary.Value` protobuf message.
///
/// ```text
/// message Summary {
///   message Value {
///     string tag          = 1;
///     float  simple_value = 2;
///   }
/// }
/// ```
fn encode_summary_value(tag: &str, value: f32) -> Vec<u8> {
    let mut pw = ProtobufWriter::new();
    pw.write_string(1, tag);
    pw.write_float(2, value);
    pw.into_bytes()
}

/// Encode a `Summary` protobuf message containing one or more values.
///
/// ```text
/// message Summary {
///   repeated Value value = 1;
/// }
/// ```
fn encode_summary(values: &[(&str, f32)]) -> Vec<u8> {
    let mut pw = ProtobufWriter::new();
    for &(tag, value) in values {
        let val_bytes = encode_summary_value(tag, value);
        pw.write_message(1, &val_bytes);
    }
    pw.into_bytes()
}

/// Encode an `Event` protobuf message.
///
/// ```text
/// message Event {
///   double  wall_time    = 1;
///   int64   step         = 2;
///   // oneof what {
///   string  file_version = 3;
///   Summary summary      = 5;
///   // }
/// }
/// ```
fn encode_event_summary(wall_time: f64, step: i64, summary_bytes: &[u8]) -> Vec<u8> {
    let mut pw = ProtobufWriter::new();
    pw.write_double(1, wall_time);
    pw.write_int64(2, step);
    pw.write_message(5, summary_bytes);
    pw.into_bytes()
}

/// Encode a `file_version` event (the first record in every TFEvents file).
fn encode_event_file_version(wall_time: f64) -> Vec<u8> {
    let mut pw = ProtobufWriter::new();
    pw.write_double(1, wall_time);
    pw.write_int64(2, 0);
    pw.write_string(3, "brain.Event:2");
    pw.into_bytes()
}

// ===========================================================================
// TFEvents record framing
// ===========================================================================

/// Write a single TFEvents record to `writer`.
///
/// Record format:
/// ```text
/// [uint64 length][uint32 masked_crc32c(length)][bytes data][uint32 masked_crc32c(data)]
/// ```
fn write_record(writer: &mut impl Write, data: &[u8]) -> FerrotorchResult<()> {
    let len = data.len() as u64;
    let len_bytes = len.to_le_bytes();
    let len_crc = masked_crc32c(&len_bytes);
    let data_crc = masked_crc32c(data);

    writer
        .write_all(&len_bytes)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("TensorBoard write failed: {e}"),
        })?;
    writer
        .write_all(&len_crc.to_le_bytes())
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("TensorBoard write failed: {e}"),
        })?;
    writer
        .write_all(data)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("TensorBoard write failed: {e}"),
        })?;
    writer
        .write_all(&data_crc.to_le_bytes())
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("TensorBoard write failed: {e}"),
        })?;

    Ok(())
}

// ===========================================================================
// TensorBoardWriter
// ===========================================================================

/// Writes TensorBoard-compatible event files.
///
/// Creates a TFEvents file inside `log_dir` and appends scalar summaries to
/// it. The resulting file can be visualized with:
///
/// ```bash
/// tensorboard --logdir=runs/
/// ```
///
/// # Example
///
/// ```ignore
/// use ferrotorch_train::TensorBoardWriter;
///
/// let mut writer = TensorBoardWriter::new("runs/experiment_1")?;
/// for step in 0..100 {
///     writer.add_scalar("loss/train", loss_value, step)?;
/// }
/// writer.flush()?;
/// ```
pub struct TensorBoardWriter {
    log_dir: PathBuf,
    file: BufWriter<File>,
    step: i64,
}

impl TensorBoardWriter {
    /// Create a new writer that writes events to `log_dir`.
    ///
    /// The directory is created if it does not exist. A TFEvents file is
    /// created inside it, prefixed with the current wall-clock timestamp.
    /// A `file_version` event is written as the first record.
    ///
    /// # Errors
    ///
    /// Returns `FerrotorchError::InvalidArgument` when:
    /// - `log_dir` cannot be created (e.g. permission denied, parent missing).
    /// - The event file cannot be created at the chosen path.
    /// - Writing the initial `file_version` record to the new file fails.
    pub fn new(log_dir: impl AsRef<Path>) -> FerrotorchResult<Self> {
        let log_dir = log_dir.as_ref().to_path_buf();
        fs::create_dir_all(&log_dir).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to create TensorBoard log directory: {e}"),
        })?;

        let wall_time = wall_time_secs();
        let timestamp = wall_time as u64;
        let filename = format!("events.out.tfevents.{timestamp}.ferrotorch");
        let path = log_dir.join(&filename);

        let file = File::create(&path).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to create TensorBoard event file: {e}"),
        })?;
        let mut file = BufWriter::new(file);

        // Write the mandatory file_version record.
        let version_event = encode_event_file_version(wall_time);
        write_record(&mut file, &version_event)?;

        Ok(Self {
            log_dir,
            file,
            step: 0,
        })
    }

    /// Add a single scalar value at the given step.
    ///
    /// # Errors
    ///
    /// Returns `FerrotorchError::InvalidArgument` when the underlying
    /// buffered writer fails (e.g. disk full, broken pipe, closed file).
    pub fn add_scalar(&mut self, tag: &str, value: f64, step: i64) -> FerrotorchResult<()> {
        let wall_time = wall_time_secs();
        let summary = encode_summary(&[(tag, value as f32)]);
        let event = encode_event_summary(wall_time, step, &summary);
        write_record(&mut self.file, &event)?;
        self.step = step;
        Ok(())
    }

    /// Add multiple scalar values under a common prefix at the given step.
    ///
    /// Each entry in `values` is written as `"{main_tag}/{sub_tag}"`.
    ///
    /// # Errors
    ///
    /// Returns `FerrotorchError::InvalidArgument` when the underlying
    /// buffered writer fails (e.g. disk full, broken pipe, closed file).
    pub fn add_scalars(
        &mut self,
        main_tag: &str,
        values: &HashMap<String, f64>,
        step: i64,
    ) -> FerrotorchResult<()> {
        let wall_time = wall_time_secs();
        let pairs: Vec<(String, f32)> = values
            .iter()
            .map(|(k, &v)| (format!("{main_tag}/{k}"), v as f32))
            .collect();
        let tag_value_refs: Vec<(&str, f32)> =
            pairs.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let summary = encode_summary(&tag_value_refs);
        let event = encode_event_summary(wall_time, step, &summary);
        write_record(&mut self.file, &event)?;
        self.step = step;
        Ok(())
    }

    /// Flush buffered data to the underlying file.
    ///
    /// # Errors
    ///
    /// Returns `FerrotorchError::InvalidArgument` when the underlying
    /// `BufWriter::flush` fails (e.g. disk full, broken pipe, closed file).
    pub fn flush(&mut self) -> FerrotorchResult<()> {
        self.file
            .flush()
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("TensorBoard flush failed: {e}"),
            })
    }

    /// Return the log directory path.
    pub fn log_dir(&self) -> &Path {
        &self.log_dir
    }
}

/// Get the current wall-clock time in seconds since the Unix epoch.
fn wall_time_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

// ===========================================================================
// TensorBoardCallback
// ===========================================================================

/// A [`Callback`] that logs training metrics to TensorBoard.
///
/// Writes `train_loss`, `val_loss`, and `lr` as scalars at the end of each
/// epoch, and per-batch loss at the end of each batch.
///
/// # Example
///
/// ```ignore
/// use ferrotorch_train::{Learner, TensorBoardCallback};
///
/// let tb = TensorBoardCallback::new("runs/experiment_1")?;
/// let mut learner = Learner::new(model, optimizer, loss_fn)
///     .with_callback(Box::new(tb));
/// ```
pub struct TensorBoardCallback {
    writer: Mutex<TensorBoardWriter>,
}

impl TensorBoardCallback {
    /// Create a new callback that writes events to `log_dir`.
    ///
    /// # Errors
    ///
    /// Returns `FerrotorchError::InvalidArgument` for any reason the
    /// underlying [`TensorBoardWriter::new`] would: the log directory
    /// cannot be created, the event file cannot be created, or the
    /// `file_version` record cannot be written.
    pub fn new(log_dir: impl AsRef<Path>) -> FerrotorchResult<Self> {
        let writer = TensorBoardWriter::new(log_dir)?;
        Ok(Self {
            writer: Mutex::new(writer),
        })
    }
}

impl<T: Float> Callback<T> for TensorBoardCallback {
    fn on_epoch_end(&mut self, epoch: usize, result: &EpochResult) {
        let mut writer = match self.writer.lock() {
            Ok(w) => w,
            Err(poisoned) => poisoned.into_inner(),
        };
        let step = epoch as i64;

        // PyTorch's `torch.utils.tensorboard.SummaryWriter` logs but does
        // not raise when an `add_scalar` write fails (e.g. disk full,
        // closed file handle); the `Callback` trait surface returns `()`,
        // so we mirror that policy and emit a `tracing::warn!` per failed
        // write instead of swallowing the error silently.
        if let Err(e) = writer.add_scalar("train_loss", result.train_loss, step) {
            tracing::warn!(
                target: "ferrotorch::tensorboard",
                error = %e,
                kind = "train_loss",
                step,
                "TensorBoard write failed",
            );
        }

        if let Some(val_loss) = result.val_loss
            && let Err(e) = writer.add_scalar("val_loss", val_loss, step)
        {
            tracing::warn!(
                target: "ferrotorch::tensorboard",
                error = %e,
                kind = "val_loss",
                step,
                "TensorBoard write failed",
            );
        }

        if let Err(e) = writer.add_scalar("lr", result.lr, step) {
            tracing::warn!(
                target: "ferrotorch::tensorboard",
                error = %e,
                kind = "lr",
                step,
                "TensorBoard write failed",
            );
        }

        // Write any custom metrics.
        for (name, &value) in &result.metrics {
            if let Err(e) = writer.add_scalar(name, value, step) {
                tracing::warn!(
                    target: "ferrotorch::tensorboard",
                    error = %e,
                    kind = "metric",
                    metric = %name,
                    step,
                    "TensorBoard write failed",
                );
            }
        }

        if let Err(e) = writer.flush() {
            tracing::warn!(
                target: "ferrotorch::tensorboard",
                error = %e,
                kind = "flush",
                step,
                "TensorBoard flush failed",
            );
        }
    }

    fn on_batch_end(&mut self, batch: usize, loss: f64) {
        let mut writer = match self.writer.lock() {
            Ok(w) => w,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Err(e) = writer.add_scalar("batch_loss", loss, batch as i64) {
            tracing::warn!(
                target: "ferrotorch::tensorboard",
                error = %e,
                kind = "batch_loss",
                batch,
                "TensorBoard write failed",
            );
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::history::EpochResult;

    /// CRC32C of the empty input must be zero.
    #[test]
    fn test_crc32c_empty() {
        assert_eq!(crc32c(b""), 0x0000_0000);
    }

    /// RFC 3720 test vector: CRC32C of 32 bytes of zeros is 0x8A9136AA.
    #[test]
    fn test_crc32c_zeros() {
        let data = [0u8; 32];
        assert_eq!(crc32c(&data), 0x8A91_36AA);
    }

    /// RFC 3720 test vector: CRC32C of 32 bytes of 0xFF is 0x62A8AB43.
    #[test]
    fn test_crc32c_ones() {
        let data = [0xFFu8; 32];
        assert_eq!(crc32c(&data), 0x62A8_AB43);
    }

    /// RFC 3720 test vector: CRC32C of 32 incrementing bytes is 0x46DD794E.
    #[test]
    fn test_crc32c_incrementing() {
        let data: Vec<u8> = (0u8..32).collect();
        assert_eq!(crc32c(&data), 0x46DD_794E);
    }

    /// Masked CRC32C should differ from raw CRC32C.
    #[test]
    fn test_masked_crc32c_differs_from_raw() {
        let data = b"hello";
        assert_ne!(crc32c(data), masked_crc32c(data));
    }

    /// Writing a scalar creates a non-empty event file.
    #[test]
    fn test_writer_creates_file() {
        let dir = tempdir("tb_test_creates_file");
        let mut writer = TensorBoardWriter::new(&dir).unwrap();
        writer.add_scalar("loss", 0.5, 0).unwrap();
        writer.flush().unwrap();

        let entries: Vec<_> = fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).collect();
        assert_eq!(entries.len(), 1);

        let metadata = entries[0].metadata().unwrap();
        assert!(metadata.len() > 0, "event file should be non-empty");
    }

    /// Writing multiple scalars at different steps produces a growing file.
    #[test]
    fn test_writer_multiple_scalars() {
        let dir = tempdir("tb_test_multi_scalar");
        let mut writer = TensorBoardWriter::new(&dir).unwrap();
        writer.add_scalar("loss", 1.0, 0).unwrap();
        writer.flush().unwrap();

        let size_after_one = file_size(&dir);

        writer.add_scalar("loss", 0.5, 1).unwrap();
        writer.add_scalar("loss", 0.25, 2).unwrap();
        writer.flush().unwrap();

        let size_after_three = file_size(&dir);
        assert!(
            size_after_three > size_after_one,
            "file should grow with more scalars"
        );
    }

    /// The first record in the file is the `file_version` event.
    #[test]
    fn test_file_starts_with_version_event() {
        let dir = tempdir("tb_test_version");
        let writer = TensorBoardWriter::new(&dir).unwrap();
        drop(writer);

        let entries: Vec<_> = fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).collect();
        let data = fs::read(entries[0].path()).unwrap();

        // The first record must contain "brain.Event:2".
        let needle = b"brain.Event:2";
        assert!(
            data.windows(needle.len()).any(|w| w == needle),
            "first record should contain file_version string"
        );
    }

    /// `add_scalars` writes grouped scalars with a compound tag.
    #[test]
    fn test_add_scalars_grouped() {
        let dir = tempdir("tb_test_grouped");
        let mut writer = TensorBoardWriter::new(&dir).unwrap();
        let mut values = HashMap::new();
        values.insert("train".to_string(), 0.5);
        values.insert("val".to_string(), 0.6);
        writer.add_scalars("loss", &values, 0).unwrap();
        writer.flush().unwrap();

        let entries: Vec<_> = fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).collect();
        let data = fs::read(entries[0].path()).unwrap();

        // Both compound tags should appear in the binary data.
        assert!(
            data.windows(b"loss/train".len())
                .any(|w| w == b"loss/train")
        );
        assert!(data.windows(b"loss/val".len()).any(|w| w == b"loss/val"));
    }

    /// `TensorBoardCallback` implements `Callback<f32>`.
    #[test]
    fn test_tensorboard_callback_is_callback() {
        fn assert_callback<T: Callback<f32>>() {}
        assert_callback::<TensorBoardCallback>();
    }

    /// `TensorBoardCallback` is `Send + Sync`.
    #[test]
    fn test_tensorboard_callback_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TensorBoardCallback>();
    }

    /// `TensorBoardCallback::on_epoch_end` writes scalars without panicking.
    #[test]
    fn test_tensorboard_callback_epoch_end() {
        let dir = tempdir("tb_test_cb_epoch");
        let mut cb = TensorBoardCallback::new(&dir).unwrap();
        let result = EpochResult {
            epoch: 0,
            train_loss: 0.42,
            val_loss: Some(0.55),
            metrics: HashMap::new(),
            lr: 0.001,
            duration_secs: 1.0,
        };
        Callback::<f32>::on_epoch_end(&mut cb, 0, &result);

        let entries: Vec<_> = fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).collect();
        let data = fs::read(entries[0].path()).unwrap();
        assert!(
            data.windows(b"train_loss".len())
                .any(|w| w == b"train_loss")
        );
        assert!(data.windows(b"val_loss".len()).any(|w| w == b"val_loss"));
    }

    /// `TensorBoardCallback::on_batch_end` writes batch loss without panicking.
    #[test]
    fn test_tensorboard_callback_batch_end() {
        let dir = tempdir("tb_test_cb_batch");
        let mut cb = TensorBoardCallback::new(&dir).unwrap();
        Callback::<f32>::on_batch_end(&mut cb, 0, 0.99);

        // Flush the internal writer so buffered data reaches disk.
        cb.writer.lock().unwrap().flush().unwrap();

        let entries: Vec<_> = fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).collect();
        let data = fs::read(entries[0].path()).unwrap();
        assert!(
            data.windows(b"batch_loss".len())
                .any(|w| w == b"batch_loss")
        );
    }

    /// Protobuf encoding round-trip: an event's raw bytes should contain the
    /// expected tag string.
    #[test]
    #[allow(clippy::approx_constant)] // 3.14 is an arbitrary metric value, not π.
    fn test_protobuf_encoding_contains_tag() {
        let summary = encode_summary(&[("my_metric", 3.14)]);
        assert!(
            summary
                .windows(b"my_metric".len())
                .any(|w| w == b"my_metric")
        );

        let event = encode_event_summary(0.0, 0, &summary);
        assert!(event.windows(b"my_metric".len()).any(|w| w == b"my_metric"));
    }

    /// The file_version event encodes the expected version string.
    #[test]
    fn test_file_version_event_contains_string() {
        let event = encode_event_file_version(0.0);
        assert!(
            event
                .windows(b"brain.Event:2".len())
                .any(|w| w == b"brain.Event:2")
        );
    }

    /// Record framing: a written record has expected minimum size.
    ///
    /// For data of length N, the record is 8 (len) + 4 (len_crc) + N + 4 (data_crc) = N + 16.
    #[test]
    fn test_record_framing_size() {
        let data = b"hello";
        let mut buf = Vec::new();
        write_record(&mut buf, data).unwrap();
        assert_eq!(buf.len(), data.len() + 16);
    }

    /// Record framing: the first 8 bytes encode the data length as little-endian u64.
    #[test]
    fn test_record_framing_length_field() {
        let data = b"hello";
        let mut buf = Vec::new();
        write_record(&mut buf, data).unwrap();
        let stored_len = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        assert_eq!(stored_len, data.len() as u64);
    }

    // -- helpers ------------------------------------------------------------

    /// Create a temporary directory for test output.
    fn tempdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ferrotorch_tb_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Get the total size of all files in a directory.
    fn file_size(dir: &Path) -> u64 {
        fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum()
    }
}
