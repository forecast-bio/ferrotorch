//! `ImageFolder` and `DatasetFolder` for class-per-directory datasets. (#589)
//!
//! Mirrors `torchvision.datasets.ImageFolder` / `DatasetFolder`. The root
//! directory is expected to contain one subdirectory per class:
//!
//! ```text
//! root/
//!   cat/
//!     0001.jpg
//!     0002.jpg
//!   dog/
//!     0001.jpg
//!     ...
//! ```
//!
//! The class list is the alphabetically-sorted list of subdirectory names,
//! and each class is assigned a stable integer index in that order.
//! Within a class, files are sorted alphabetically too — load order is
//! deterministic across runs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};
use ferrotorch_data::Dataset;

/// One sample produced by [`ImageFolder`]: a CHW image tensor + class index.
///
/// Marked `#[non_exhaustive]` so future per-sample metadata (e.g. file
/// path) can be added without breaking struct-literal construction outside
/// this crate.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ImageSample<T: Float> {
    /// Image tensor with shape `[C, H, W]`, values in `[0, 1]`.
    pub image: Tensor<T>,
    /// Integer class index in `0..num_classes`.
    pub label: u32,
}

/// File extensions accepted by [`ImageFolder`] when scanning a class
/// directory. Comparison is case-insensitive.
pub const IMG_EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp",
];

/// Class-per-subdirectory image dataset. `from_dir` walks the root and
/// builds `(path, class_idx)` samples, then [`Dataset::get`] reads each
/// image lazily via [`crate::io::read_image_as_tensor`].
pub struct ImageFolder<T: Float> {
    samples: Vec<(PathBuf, u32)>,
    classes: Vec<String>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> std::fmt::Debug for ImageFolder<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageFolder")
            .field("num_samples", &self.samples.len())
            .field("classes", &self.classes)
            .finish()
    }
}

impl<T: Float> ImageFolder<T> {
    /// Walk `root`, discover one class per subdirectory, and collect every
    /// image file with a known extension. Returns an empty dataset if no
    /// classes are found (matches torchvision).
    pub fn from_dir(root: impl AsRef<Path>) -> FerrotorchResult<Self> {
        Self::from_dir_with_extensions(root, IMG_EXTENSIONS)
    }

    /// Variant of [`from_dir`] with a caller-supplied extension list.
    /// Pass an empty slice to accept all files.
    pub fn from_dir_with_extensions(
        root: impl AsRef<Path>,
        extensions: &[&str],
    ) -> FerrotorchResult<Self> {
        let (samples, classes) = scan_class_dirs(root.as_ref(), extensions, |_| true)?;
        Ok(Self {
            samples,
            classes,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Variant of [`from_dir`] with a custom predicate. The predicate is
    /// applied **after** the extension filter — return `false` to drop
    /// files that pass the extension check but should be excluded
    /// (e.g. corrupt thumbnails, deny-listed paths).
    pub fn from_dir_with_filter<F: Fn(&Path) -> bool>(
        root: impl AsRef<Path>,
        is_valid_file: F,
    ) -> FerrotorchResult<Self> {
        let (samples, classes) = scan_class_dirs(root.as_ref(), IMG_EXTENSIONS, is_valid_file)?;
        Ok(Self {
            samples,
            classes,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Sorted list of class names (subdirectory basenames).
    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    /// Number of distinct classes.
    pub fn num_classes(&self) -> usize {
        self.classes.len()
    }

    /// Map from class name to integer index (the index used in `Sample::label`).
    pub fn class_to_idx(&self) -> HashMap<&str, u32> {
        self.classes
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i as u32))
            .collect()
    }

    /// The full `(path, class_idx)` list. Order matches `Dataset::get` indices.
    pub fn samples(&self) -> &[(PathBuf, u32)] {
        &self.samples
    }
}

impl<T: Float + 'static> Dataset for ImageFolder<T> {
    type Sample = ImageSample<T>;

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> FerrotorchResult<Self::Sample> {
        let (path, label) = self
            .samples
            .get(index)
            .ok_or(FerrotorchError::InvalidArgument {
                message: format!(
                    "ImageFolder::get: index {index} out of range (len={})",
                    self.samples.len()
                ),
            })?;
        let image = crate::io::read_image_as_tensor::<T>(path)?;
        Ok(ImageSample {
            image,
            label: *label,
        })
    }
}

/// Generalized class-per-subdirectory dataset. The loader function decides
/// how each file becomes a sample (audio decode, custom binary format,
/// etc.). Mirrors `torchvision.datasets.DatasetFolder`.
pub struct DatasetFolder<S, F: Fn(&Path) -> FerrotorchResult<S>> {
    samples: Vec<(PathBuf, u32)>,
    classes: Vec<String>,
    loader: F,
}

/// Sample produced by [`DatasetFolder`]: the loader's output + class index.
#[derive(Debug, Clone)]
pub struct FolderSample<S> {
    /// Whatever the loader returned for this file.
    pub data: S,
    /// Integer class index in `0..num_classes`.
    pub label: u32,
}

// `loader` is a generic `Fn` closure that doesn't implement `Debug`; we
// intentionally omit it from the printed representation.
#[allow(clippy::missing_fields_in_debug)]
impl<S, F> std::fmt::Debug for DatasetFolder<S, F>
where
    F: Fn(&Path) -> FerrotorchResult<S>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DatasetFolder")
            .field("num_samples", &self.samples.len())
            .field("classes", &self.classes)
            .finish()
    }
}

impl<S, F: Fn(&Path) -> FerrotorchResult<S>> DatasetFolder<S, F> {
    /// Walk `root` and accept files whose extension (case-insensitive) is
    /// in `extensions`. The `loader` is stored and invoked by `Dataset::get`.
    pub fn from_dir(
        root: impl AsRef<Path>,
        extensions: &[&str],
        loader: F,
    ) -> FerrotorchResult<Self> {
        let (samples, classes) = scan_class_dirs(root.as_ref(), extensions, |_| true)?;
        Ok(Self {
            samples,
            classes,
            loader,
        })
    }

    /// Variant with both an extension list and a custom predicate.
    pub fn from_dir_with_filter<P: Fn(&Path) -> bool>(
        root: impl AsRef<Path>,
        extensions: &[&str],
        is_valid_file: P,
        loader: F,
    ) -> FerrotorchResult<Self> {
        let (samples, classes) = scan_class_dirs(root.as_ref(), extensions, is_valid_file)?;
        Ok(Self {
            samples,
            classes,
            loader,
        })
    }

    /// Sorted class names.
    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    /// Number of classes.
    pub fn num_classes(&self) -> usize {
        self.classes.len()
    }

    /// Sample list — `(path, class_idx)` pairs.
    pub fn samples(&self) -> &[(PathBuf, u32)] {
        &self.samples
    }
}

impl<S, F> Dataset for DatasetFolder<S, F>
where
    S: Send + 'static,
    F: Fn(&Path) -> FerrotorchResult<S> + Send + Sync + 'static,
{
    type Sample = FolderSample<S>;

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> FerrotorchResult<Self::Sample> {
        let (path, label) = self
            .samples
            .get(index)
            .ok_or(FerrotorchError::InvalidArgument {
                message: format!(
                    "DatasetFolder::get: index {index} out of range (len={})",
                    self.samples.len()
                ),
            })?;
        let data = (self.loader)(path)?;
        Ok(FolderSample {
            data,
            label: *label,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal: directory walk
// ---------------------------------------------------------------------------

/// Walk `root` and produce `(samples, classes)`. Subdirectories are sorted
/// alphabetically; files within each subdirectory are sorted too. Hidden
/// dotfiles are skipped to match torchvision behavior.
#[allow(clippy::type_complexity)]
fn scan_class_dirs<F: Fn(&Path) -> bool>(
    root: &Path,
    extensions: &[&str],
    is_valid_file: F,
) -> FerrotorchResult<(Vec<(PathBuf, u32)>, Vec<String>)> {
    if !root.is_dir() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "ImageFolder/DatasetFolder: root path is not a directory: {}",
                root.display()
            ),
        });
    }

    // Collect class subdirectories.
    let mut class_dirs: Vec<(String, PathBuf)> = Vec::new();
    let entries = std::fs::read_dir(root).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to read directory {}: {e}", root.display()),
    })?;
    for entry in entries {
        let entry = entry.map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to iterate {}: {e}", root.display()),
        })?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        class_dirs.push((name.to_string(), path));
    }
    class_dirs.sort_by(|a, b| a.0.cmp(&b.0));
    let classes: Vec<String> = class_dirs.iter().map(|(name, _)| name.clone()).collect();

    let mut samples: Vec<(PathBuf, u32)> = Vec::new();
    for (idx, (_class_name, class_dir)) in class_dirs.iter().enumerate() {
        let class_idx = idx as u32;
        let dir_iter =
            std::fs::read_dir(class_dir).map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("failed to read class dir {}: {e}", class_dir.display()),
            })?;
        let mut files: Vec<PathBuf> = Vec::new();
        for entry in dir_iter {
            let entry = entry.map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("failed to iterate {}: {e}", class_dir.display()),
            })?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with('.') {
                    continue;
                }
            }
            if !extensions.is_empty() && !has_extension_ci(&path, extensions) {
                continue;
            }
            if !is_valid_file(&path) {
                continue;
            }
            files.push(path);
        }
        files.sort();
        for path in files {
            samples.push((path, class_idx));
        }
    }
    Ok((samples, classes))
}

/// Case-insensitive extension match.
fn has_extension_ci(path: &Path, extensions: &[&str]) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    let ext_lower = ext.to_ascii_lowercase();
    extensions
        .iter()
        .any(|e| e.eq_ignore_ascii_case(&ext_lower))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tempdir with subdirectories `cat/`, `dog/`, `bird/` each
    /// containing N files. PNG files are written as real 2×2 grayscale PNGs
    /// via the `image` crate; other extensions just get placeholder bytes
    /// (the folder scan only reads filenames, so non-PNG tests don't need
    /// decode-able payloads).
    fn make_image_tree(extensions: &[&str], files_per_class: usize) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        for class in &["cat", "dog", "bird"] {
            let dir = tmp.path().join(class);
            std::fs::create_dir(&dir).unwrap();
            for i in 0..files_per_class {
                for ext in extensions {
                    let path = dir.join(format!("{i:04}.{ext}"));
                    if ext == &"png" {
                        let img =
                            image::GrayImage::from_raw(2, 2, vec![0u8, 64, 128, 255]).unwrap();
                        img.save(&path).unwrap();
                    } else {
                        std::fs::write(&path, b"placeholder").unwrap();
                    }
                }
            }
        }
        tmp
    }

    #[test]
    fn image_folder_discovers_classes_in_alphabetical_order() {
        let tmp = make_image_tree(&["jpg"], 2);
        let folder = ImageFolder::<f32>::from_dir(tmp.path()).unwrap();
        assert_eq!(folder.classes(), &["bird", "cat", "dog"]);
        assert_eq!(folder.num_classes(), 3);
        assert_eq!(folder.class_to_idx()["bird"], 0);
        assert_eq!(folder.class_to_idx()["cat"], 1);
        assert_eq!(folder.class_to_idx()["dog"], 2);
    }

    #[test]
    fn image_folder_collects_all_files_per_class() {
        let tmp = make_image_tree(&["jpg"], 5);
        let folder = ImageFolder::<f32>::from_dir(tmp.path()).unwrap();
        assert_eq!(folder.len(), 3 * 5);
        // First 5 are `bird`, next 5 are `cat`, last 5 are `dog`.
        for (i, (_path, label)) in folder.samples().iter().enumerate() {
            let expected = (i / 5) as u32;
            assert_eq!(*label, expected);
        }
    }

    #[test]
    fn image_folder_filters_unknown_extensions() {
        let tmp = tempfile::tempdir().unwrap();
        let cat_dir = tmp.path().join("cat");
        std::fs::create_dir(&cat_dir).unwrap();
        std::fs::write(cat_dir.join("good.jpg"), b"x").unwrap();
        std::fs::write(cat_dir.join("ignore.txt"), b"x").unwrap();
        std::fs::write(cat_dir.join("ignore.tmp"), b"x").unwrap();

        let folder = ImageFolder::<f32>::from_dir(tmp.path()).unwrap();
        assert_eq!(folder.len(), 1);
        assert!(
            folder.samples()[0]
                .0
                .to_str()
                .unwrap()
                .ends_with("good.jpg")
        );
    }

    #[test]
    fn image_folder_extension_matching_is_case_insensitive() {
        let tmp = tempfile::tempdir().unwrap();
        let cat_dir = tmp.path().join("cat");
        std::fs::create_dir(&cat_dir).unwrap();
        std::fs::write(cat_dir.join("upper.JPG"), b"x").unwrap();
        std::fs::write(cat_dir.join("mixed.PnG"), b"x").unwrap();

        let folder = ImageFolder::<f32>::from_dir(tmp.path()).unwrap();
        assert_eq!(folder.len(), 2);
    }

    #[test]
    fn image_folder_skips_dotfiles_and_dot_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let cat_dir = tmp.path().join("cat");
        std::fs::create_dir(&cat_dir).unwrap();
        std::fs::write(cat_dir.join(".DS_Store"), b"x").unwrap();
        std::fs::write(cat_dir.join("good.jpg"), b"x").unwrap();
        std::fs::create_dir(tmp.path().join(".hidden_class")).unwrap();
        std::fs::write(tmp.path().join(".hidden_class/file.jpg"), b"x").unwrap();

        let folder = ImageFolder::<f32>::from_dir(tmp.path()).unwrap();
        assert_eq!(folder.classes(), &["cat"]);
        assert_eq!(folder.len(), 1);
    }

    #[test]
    fn image_folder_with_filter_drops_predicate_rejects() {
        let tmp = make_image_tree(&["jpg"], 4);
        // Reject every file whose stem ends in an even digit.
        let folder = ImageFolder::<f32>::from_dir_with_filter(tmp.path(), |p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .map(|s| {
                    s.chars()
                        .last()
                        .and_then(|c| c.to_digit(10))
                        .map(|d| d % 2 == 1)
                        .unwrap_or(true)
                })
                .unwrap_or(true)
        })
        .unwrap();
        // 4 files per class, half kept → 6.
        assert_eq!(folder.len(), 6);
    }

    #[test]
    fn image_folder_get_reads_real_image() {
        let tmp = make_image_tree(&["png"], 1);
        let folder = ImageFolder::<f32>::from_dir(tmp.path()).unwrap();
        assert_eq!(folder.len(), 3);
        let sample = folder.get(0).unwrap();
        // 2×2 grayscale PNG → CHW with H=W=2.
        let shape = sample.image.shape();
        assert_eq!(shape.len(), 3);
        assert_eq!(shape[1], 2);
        assert_eq!(shape[2], 2);
        assert_eq!(sample.label, 0); // bird
    }

    #[test]
    fn image_folder_get_out_of_range_errors() {
        let tmp = make_image_tree(&["jpg"], 1);
        let folder = ImageFolder::<f32>::from_dir(tmp.path()).unwrap();
        let err = folder.get(100).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn image_folder_rejects_non_directory_root() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let err = ImageFolder::<f32>::from_dir(tmp.path()).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn dataset_folder_uses_custom_loader() {
        // Build a directory of plain-text "files"; the loader returns the
        // file's byte length. This tests that DatasetFolder is generic
        // over the sample type.
        let tmp = tempfile::tempdir().unwrap();
        for (i, class) in ["short", "long"].iter().enumerate() {
            let dir = tmp.path().join(class);
            std::fs::create_dir(&dir).unwrap();
            for j in 0..3 {
                let n = if i == 0 { 5 } else { 50 };
                std::fs::write(dir.join(format!("{j}.txt")), vec![b'x'; n]).unwrap();
            }
        }

        let ds: DatasetFolder<usize, _> =
            DatasetFolder::from_dir(tmp.path(), &["txt"], |p: &Path| {
                let bytes = std::fs::read(p).map_err(|e| FerrotorchError::InvalidArgument {
                    message: format!("read failed: {e}"),
                })?;
                Ok(bytes.len())
            })
            .unwrap();
        assert_eq!(ds.len(), 6);
        // class 0 is "long" alphabetically, class 1 is "short".
        for i in 0..6 {
            let s = ds.get(i).unwrap();
            if s.label == 0 {
                assert_eq!(s.data, 50);
            } else {
                assert_eq!(s.data, 5);
            }
        }
    }

    #[test]
    fn dataset_folder_with_no_extensions_accepts_all_files() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("any");
        std::fs::create_dir(&dir).unwrap();
        std::fs::write(dir.join("a.foo"), b"a").unwrap();
        std::fs::write(dir.join("b.bar"), b"b").unwrap();
        std::fs::write(dir.join("c"), b"c").unwrap();

        let ds: DatasetFolder<u8, _> = DatasetFolder::from_dir(tmp.path(), &[], |p: &Path| {
            let bytes = std::fs::read(p).map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("read failed: {e}"),
            })?;
            Ok(bytes[0])
        })
        .unwrap();
        assert_eq!(ds.len(), 3);
    }
}
