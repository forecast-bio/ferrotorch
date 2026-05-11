//! Whisper audio preprocessing — 16 kHz mono f32 PCM → `[1, 80, 3000]`
//! log-mel spectrogram.
//!
//! This is a direct port of `transformers.WhisperFeatureExtractor` (the
//! NumPy path through `transformers.audio_utils.spectrogram` with
//! `power=2.0`, `log_mel="log10"`, `mel_floor=1e-10`, and Whisper's
//! 80-bin Slaney-derived mel filter bank). The 80 × 201 filter bank is
//! shipped as the binary asset `assets/mel_filters_80x201.bin` — bytes
//! generated from `WhisperFeatureExtractor.mel_filters.T.astype("<f4")`
//! so we match the reference filterbank byte-for-byte rather than
//! re-deriving the mel scale.
//!
//! The pipeline:
//!
//! 1. Pad / trim audio to `chunk_length × sample_rate = 30 × 16_000 = 480 000`
//!    samples (`pad_or_trim`).
//! 2. STFT with `n_fft=400`, `hop_length=160`, periodic Hann window,
//!    `center=True` (reflect-pad by `n_fft / 2 = 200`) → 3001 frames of
//!    201 complex bins.
//! 3. Magnitude-squared (`power=2.0`) → `[201, 3001]` real.
//! 4. Drop the last frame (matches `log_spec[:, :-1]` in the reference) →
//!    `[201, 3000]`.
//! 5. Apply 80-mel filter bank (`mel @ spec` → `[80, 3000]`) with
//!    floor `1e-10`.
//! 6. `log10(x)` then `clip(x, x.max() - 8.0)` then `(x + 4.0) / 4.0`.
//! 7. Promote to `[1, 80, 3000]` `Tensor<f32>`.
//!
//! All math is done in `f64` accumulation and downcast at the end to
//! match the reference's `np.float64` STFT path.

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor, TensorStorage};

/// Whisper sampling rate (16 kHz).
pub const SAMPLE_RATE: u32 = 16_000;
/// Whisper chunk length in seconds (30 s).
pub const CHUNK_LENGTH: usize = 30;
/// FFT size / analysis-frame length (25 ms at 16 kHz).
pub const N_FFT: usize = 400;
/// STFT hop size (10 ms at 16 kHz).
pub const HOP_LENGTH: usize = 160;
/// Number of mel bins.
pub const N_MELS: usize = 80;
/// Number of frames the encoder consumes (= `chunk_length × 100 fps`).
pub const N_FRAMES: usize = 3_000;
/// Number of audio samples in a fully-padded chunk
/// (= `chunk_length × sample_rate`).
pub const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE as usize;
/// Number of non-redundant rfft bins for `n_fft = 400` → `201`.
pub const N_FREQS: usize = N_FFT / 2 + 1;

/// 80-mel filter bank bytes. Layout: `[N_MELS, N_FREQS] = [80, 201]` row-major
/// little-endian `f32`. Generated from
/// `WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny").mel_filters.T`
/// (i.e. the transpose of the `[201, 80]` filter the upstream feature
/// extractor stores). Byte-for-byte identical to the reference filter so
/// any drift between this module and the reference cannot be blamed on
/// a different mel scale.
const MEL_FILTERS_BYTES: &[u8] =
    include_bytes!("../assets/mel_filters_80x201.bin");

/// Decode the embedded `[80, 201]` mel filter bank into an owned `Vec<f32>`.
fn mel_filters() -> Vec<f32> {
    let n = N_MELS * N_FREQS;
    let expected_bytes = n * 4;
    assert_eq!(
        MEL_FILTERS_BYTES.len(),
        expected_bytes,
        "mel filter asset size mismatch: got {} bytes, expected {expected_bytes}",
        MEL_FILTERS_BYTES.len(),
    );
    let mut out = Vec::with_capacity(n);
    for chunk in MEL_FILTERS_BYTES.chunks_exact(4) {
        let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
        out.push(f32::from_le_bytes(arr));
    }
    out
}

/// Pad-or-trim `audio` to exactly `N_SAMPLES` (480 000) samples.
///
/// Pad with trailing zeros if shorter (`np.pad(audio, (0, N_SAMPLES - len))`),
/// or take the leading prefix if longer.
fn pad_or_trim(audio: &[f32]) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::with_capacity(N_SAMPLES);
    if audio.len() >= N_SAMPLES {
        out.extend(audio[..N_SAMPLES].iter().map(|&v| v as f64));
    } else {
        out.extend(audio.iter().map(|&v| v as f64));
        out.resize(N_SAMPLES, 0.0);
    }
    out
}

/// Periodic Hann window of length `N_FFT`, matching
/// `transformers.audio_utils.window_function(N_FFT, "hann", periodic=True)`:
/// build `length = N_FFT + 1` samples of the symmetric formula then drop
/// the trailing one. Each sample is
/// `0.5 - 0.5 * cos(2π k / N_FFT)` for `k = 0..N_FFT`.
fn hann_window() -> Vec<f64> {
    let length = N_FFT + 1;
    let mut window: Vec<f64> = Vec::with_capacity(N_FFT);
    let denom = (length - 1) as f64;
    for k in 0..length {
        let v = 0.5 - 0.5 * ((2.0 * std::f64::consts::PI * k as f64) / denom).cos();
        if k < N_FFT {
            window.push(v);
        }
    }
    // For `periodic=True`, `length = N_FFT + 1`, and we drop the last
    // sample — but the reference *zeros-pads* the dropped slot back to
    // `N_FFT` (the window slot stays sized `length` and the last entry
    // is implicitly the period-end). We already pushed exactly `N_FFT`
    // samples above; that matches `np.hanning(N_FFT+1)[:-1]` (numpy's
    // 0-edge convention).
    window
}

/// Reflect-pad `audio` by `pad` samples on each side, matching
/// `np.pad(audio, ((pad, pad),), mode="reflect")`.
///
/// `audio[-i] = audio[i]` (the boundary sample is the axis, NOT
/// duplicated). For `pad = 200` and 480 000 samples we get a 480 400
/// sample buffer.
fn reflect_pad(audio: &[f64], pad: usize) -> Vec<f64> {
    let n = audio.len();
    assert!(n > pad, "reflect_pad: pad ({pad}) must be < input length ({n})");
    let mut out = Vec::with_capacity(n + 2 * pad);
    // Leading reflected pad: audio[pad], audio[pad-1], ..., audio[1]
    for i in (1..=pad).rev() {
        out.push(audio[i]);
    }
    out.extend_from_slice(audio);
    // Trailing reflected pad: audio[n-2], audio[n-3], ..., audio[n-1-pad]
    for i in 1..=pad {
        out.push(audio[n - 1 - i]);
    }
    out
}

/// One radix-2-or-better complex DFT of length `n` (Cooley-Tukey via
/// `rustfft`).
fn stft_one_frame(buffer: &[f64], out_real: &mut [f64], out_imag: &mut [f64]) {
    debug_assert_eq!(buffer.len(), N_FFT);
    debug_assert_eq!(out_real.len(), N_FREQS);
    debug_assert_eq!(out_imag.len(), N_FREQS);
    // Naive DFT of length 400 (not a power of 2) is acceptable here
    // because the full feature extraction is dominated by the
    // 3001 × O(N^2) loop and we only run this once per call. A future
    // optimisation could swap to `rustfft` (added as a workspace dep).
    // For correctness we just compute the real / imag rfft directly.
    for k in 0..N_FREQS {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        let arg = -2.0 * std::f64::consts::PI * (k as f64) / (N_FFT as f64);
        for (n, &sample) in buffer.iter().enumerate() {
            let theta = arg * n as f64;
            re += sample * theta.cos();
            im += sample * theta.sin();
        }
        out_real[k] = re;
        out_imag[k] = im;
    }
}

/// Compute the `[1, 80, 3000]` log-mel spectrogram of `audio`.
///
/// `audio` is 16 kHz mono `f32` PCM in `[-1, 1]`. Sequences shorter than
/// 30 s are tail-padded with zeros; longer sequences are truncated.
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] if `audio` is empty.
/// Propagates downstream Tensor construction errors.
pub fn log_mel_spectrogram(audio: &[f32]) -> FerrotorchResult<Tensor<f32>> {
    if audio.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "log_mel_spectrogram: audio buffer is empty".into(),
        });
    }

    // -- 1. Pad / trim to 480 000 samples (30 s at 16 kHz). --------------
    let padded = pad_or_trim(audio);

    // -- 2. Reflect-pad by N_FFT / 2 = 200 (center=True). ----------------
    let reflected = reflect_pad(&padded, N_FFT / 2);

    // -- 3. Build periodic Hann window. ----------------------------------
    let window = hann_window();

    // -- 4. Frame + windowed rfft → power spectrogram. -------------------
    // num_frames = 1 + floor((len - n_fft) / hop) = 1 + (480400 - 400) / 160 = 3001.
    let total = reflected.len();
    let num_frames = 1 + (total - N_FFT) / HOP_LENGTH;
    debug_assert_eq!(num_frames, N_FRAMES + 1);
    let mut power_t = vec![0.0f64; num_frames * N_FREQS]; // [frames, freqs]
    let mut frame = vec![0.0f64; N_FFT];
    let mut re = vec![0.0f64; N_FREQS];
    let mut im = vec![0.0f64; N_FREQS];
    for f in 0..num_frames {
        let start = f * HOP_LENGTH;
        for n in 0..N_FFT {
            frame[n] = reflected[start + n] * window[n];
        }
        stft_one_frame(&frame, &mut re, &mut im);
        for k in 0..N_FREQS {
            let r = re[k];
            let i = im[k];
            power_t[f * N_FREQS + k] = r * r + i * i;
        }
    }

    // -- 5. Transpose to [freqs, frames] then drop the last frame. -------
    // power[k, t] = power_t[t, k].
    let mut power = vec![0.0f64; N_FREQS * N_FRAMES];
    for k in 0..N_FREQS {
        for t in 0..N_FRAMES {
            power[k * N_FRAMES + t] = power_t[t * N_FREQS + k];
        }
    }

    // -- 6. Apply mel filter bank with floor 1e-10. ----------------------
    // mel_filters shape: [N_MELS, N_FREQS] (row-major).
    // power shape:        [N_FREQS, N_FRAMES] (row-major).
    // out = mel @ power → [N_MELS, N_FRAMES].
    let mel = mel_filters();
    let mel_floor = 1e-10_f64;
    let mut out_mel = vec![0.0f64; N_MELS * N_FRAMES];
    for m in 0..N_MELS {
        for t in 0..N_FRAMES {
            let mut acc = 0.0f64;
            for k in 0..N_FREQS {
                acc += mel[m * N_FREQS + k] as f64 * power[k * N_FRAMES + t];
            }
            // mel_floor BEFORE log (matches `np.maximum(mel_floor, ...)`).
            out_mel[m * N_FRAMES + t] = if acc < mel_floor { mel_floor } else { acc };
        }
    }

    // -- 7. log10 → clip(max - 8) → (x + 4) / 4. -------------------------
    for v in &mut out_mel {
        *v = v.log10();
    }
    let max_val = out_mel.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let clip_at = max_val - 8.0;
    for v in &mut out_mel {
        if *v < clip_at {
            *v = clip_at;
        }
        *v = (*v + 4.0) / 4.0;
    }

    // -- 8. Downcast f64 → f32 and promote to [1, 80, 3000]. -------------
    let data: Vec<f32> = out_mel.iter().map(|&v| v as f32).collect();
    Tensor::from_storage(TensorStorage::cpu(data), vec![1, N_MELS, N_FRAMES], false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_periodic_first_last() {
        let w = hann_window();
        assert_eq!(w.len(), N_FFT);
        // periodic Hann starts at exactly 0.
        assert!(w[0].abs() < 1e-12, "Hann[0] = {} ≠ 0", w[0]);
        // last sample is one step before the period — non-zero, non-one.
        assert!(w[N_FFT - 1] > 0.0 && w[N_FFT - 1] < 1.0);
    }

    #[test]
    fn mel_filter_shape_and_norm_finite() {
        let m = mel_filters();
        assert_eq!(m.len(), N_MELS * N_FREQS);
        for &v in &m {
            assert!(v.is_finite(), "mel bank contains non-finite {v}");
        }
        // bank is non-trivial.
        let s: f32 = m.iter().sum();
        assert!(s > 0.0, "mel bank sums to zero");
    }

    #[test]
    fn reflect_pad_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r = reflect_pad(&a, 2);
        // [3, 2, 1, 2, 3, 4, 5, 4, 3]
        assert_eq!(r, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn pad_or_trim_pads_short() {
        let a = vec![0.1f32; 100];
        let p = pad_or_trim(&a);
        assert_eq!(p.len(), N_SAMPLES);
        for &v in p.iter().take(100) {
            assert!((v - 0.1).abs() < 1e-6);
        }
        for &v in p.iter().skip(100) {
            assert!(v.abs() < f64::EPSILON, "tail pad sample {v} != 0");
        }
    }

    #[test]
    fn pad_or_trim_trims_long() {
        let a = vec![0.7f32; N_SAMPLES + 100];
        let p = pad_or_trim(&a);
        assert_eq!(p.len(), N_SAMPLES);
    }

    #[test]
    fn log_mel_shape_is_1_80_3000() {
        // Silence (all zeros) is valid input; the floor kicks in.
        let audio = vec![0.0f32; N_SAMPLES];
        let mel = log_mel_spectrogram(&audio).unwrap();
        assert_eq!(mel.shape(), &[1, N_MELS, N_FRAMES]);
        for &v in mel.data().unwrap() {
            assert!(v.is_finite(), "mel non-finite: {v}");
        }
    }

    #[test]
    fn log_mel_rejects_empty() {
        let r = log_mel_spectrogram(&[]);
        assert!(r.is_err());
    }
}
