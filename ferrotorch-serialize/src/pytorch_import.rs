//! Import `PyTorch` `.pt` / `.pth` checkpoint files in pure Rust.
//!
//! A `.pt` file is a ZIP archive containing:
//! - `archive/data.pkl` (or `data.pkl`) -- pickle bytecodes describing the
//!   state dict structure (an `OrderedDict` of tensor rebuild instructions).
//! - `archive/data/0`, `archive/data/1`, ... -- raw tensor byte blobs
//!   referenced by the pickle via `PERSISTENT_LOAD`.
//!
//! This module implements:
//! 1. A minimal pickle protocol-2 parser (only the opcode subset `PyTorch` uses).
//! 2. Interpretation logic that walks the pickle tree, extracts tensor metadata
//!    (shape, dtype, storage key), and reads raw bytes from the ZIP.
//! 3. A public [`load_pytorch_state_dict`] function that produces a
//!    `StateDict<T>`.

use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::StateDict;
use memmap2::Mmap;

// ---------------------------------------------------------------------------
// Pickle value tree
// ---------------------------------------------------------------------------

/// A value produced by the pickle virtual machine.
///
/// Only the subset of types that `PyTorch`'s state dict pickles actually emit
/// is represented here. Unknown constructs are mapped to the closest
/// approximation or ignored.
#[derive(Debug, Clone)]
pub enum PickleValue {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Bytes(Vec<u8>),
    String(String),
    Tuple(Vec<PickleValue>),
    List(Vec<PickleValue>),
    Dict(Vec<(PickleValue, PickleValue)>),
    /// `GLOBAL module name` -- a Python callable / class reference.
    Global {
        module: String,
        name: String,
    },
    /// `REDUCE` -- `callable(*args)`.
    Reduce {
        callable: Box<PickleValue>,
        args: Box<PickleValue>,
    },
    /// `BUILD` -- `obj.__setstate__(state)`.
    Build {
        obj: Box<PickleValue>,
        state: Box<PickleValue>,
    },
    /// `BINPERSID` -- a persistent-load reference (used for tensor storage).
    PersistentLoad(Box<PickleValue>),
}

// ---------------------------------------------------------------------------
// Pickle opcodes (protocol 2 subset used by PyTorch)
// ---------------------------------------------------------------------------

const PROTO: u8 = 0x80;
const EMPTY_DICT: u8 = 0x7d;
const EMPTY_LIST: u8 = 0x5d;
const EMPTY_TUPLE: u8 = 0x29;
const MARK: u8 = 0x28;
const TUPLE1: u8 = 0x85;
const TUPLE2: u8 = 0x86;
const TUPLE3: u8 = 0x87;
const TUPLE: u8 = 0x74;
const BINPUT: u8 = 0x71;
const LONG_BINPUT: u8 = 0x72;
const BINGET: u8 = 0x68;
const LONG_BINGET: u8 = 0x6a;
const GLOBAL: u8 = 0x63;
const SHORT_BINUNICODE: u8 = 0x8c;
const BINUNICODE: u8 = 0x8d;
const BININT1: u8 = 0x4b;
const BININT: u8 = 0x4a;
const BININT2: u8 = 0x4d;
const BINFLOAT: u8 = 0x47;
const OP_NONE: u8 = 0x4e;
const NEWTRUE: u8 = 0x88;
const NEWFALSE: u8 = 0x89;
const REDUCE: u8 = 0x52;
const BUILD: u8 = 0x62;
const SETITEM: u8 = 0x73;
const SETITEMS: u8 = 0x75;
const APPEND: u8 = 0x61;
const APPENDS: u8 = 0x65;
const STOP: u8 = 0x2e;
const BINPERSID: u8 = 0x51;
const SHORT_BINBYTES: u8 = 0x42;
const BINBYTES: u8 = 0x44;
const SHORT_BINSTRING: u8 = 0x55;
const BINSTRING: u8 = 0x54;
const STACK_GLOBAL: u8 = 0x93;
const MEMOIZE: u8 = 0x94;
const FRAME: u8 = 0x95;
const NEWOBJ: u8 = 0x81;

// ---------------------------------------------------------------------------
// Pickle parser
// ---------------------------------------------------------------------------

/// Parse pickle bytecodes into a [`PickleValue`] tree.
///
/// This only handles the protocol-2 opcode subset that `PyTorch` state dict
/// pickles emit. Encountering an unknown opcode returns an error rather than
/// silently producing garbage.
pub fn parse_pickle(data: &[u8]) -> FerrotorchResult<PickleValue> {
    let mut pos: usize = 0;
    let mut stack: Vec<PickleValue> = Vec::new();
    let mut memo: HashMap<u32, PickleValue> = HashMap::new();
    let mut mark_stack: Vec<usize> = Vec::new();

    loop {
        if pos >= data.len() {
            return Err(pickle_err("unexpected end of pickle data"));
        }
        let opcode = data[pos];
        pos += 1;

        match opcode {
            PROTO => {
                // Skip protocol version byte.
                if pos >= data.len() {
                    return Err(pickle_err("unexpected end of pickle data (PROTO)"));
                }
                pos += 1;
            }

            FRAME => {
                // Protocol 4 framing. Skip the 8-byte frame length.
                if pos + 8 > data.len() {
                    return Err(pickle_err("unexpected end of pickle data (FRAME)"));
                }
                pos += 8;
            }

            EMPTY_DICT => {
                stack.push(PickleValue::Dict(Vec::new()));
            }

            EMPTY_LIST => {
                stack.push(PickleValue::List(Vec::new()));
            }

            EMPTY_TUPLE => {
                stack.push(PickleValue::Tuple(Vec::new()));
            }

            MARK => {
                mark_stack.push(stack.len());
            }

            TUPLE1 => {
                let a = stack.pop().ok_or_else(|| pickle_err("TUPLE1 underflow"))?;
                stack.push(PickleValue::Tuple(vec![a]));
            }

            TUPLE2 => {
                let b = stack.pop().ok_or_else(|| pickle_err("TUPLE2 underflow"))?;
                let a = stack.pop().ok_or_else(|| pickle_err("TUPLE2 underflow"))?;
                stack.push(PickleValue::Tuple(vec![a, b]));
            }

            TUPLE3 => {
                let c = stack.pop().ok_or_else(|| pickle_err("TUPLE3 underflow"))?;
                let b = stack.pop().ok_or_else(|| pickle_err("TUPLE3 underflow"))?;
                let a = stack.pop().ok_or_else(|| pickle_err("TUPLE3 underflow"))?;
                stack.push(PickleValue::Tuple(vec![a, b, c]));
            }

            TUPLE => {
                let items = pop_to_mark(&mut stack, &mut mark_stack)?;
                stack.push(PickleValue::Tuple(items));
            }

            BINPUT => {
                if pos >= data.len() {
                    return Err(pickle_err("unexpected end of pickle data (BINPUT)"));
                }
                let idx = u32::from(data[pos]);
                pos += 1;
                if let Some(top) = stack.last() {
                    memo.insert(idx, top.clone());
                }
            }

            LONG_BINPUT => {
                if pos + 4 > data.len() {
                    return Err(pickle_err("unexpected end of pickle data (LONG_BINPUT)"));
                }
                let idx =
                    u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
                pos += 4;
                if let Some(top) = stack.last() {
                    memo.insert(idx, top.clone());
                }
            }

            MEMOIZE => {
                // Protocol 4 memoize: assign next memo index to TOS.
                let idx = memo.len() as u32;
                if let Some(top) = stack.last() {
                    memo.insert(idx, top.clone());
                }
            }

            BINGET => {
                if pos >= data.len() {
                    return Err(pickle_err("unexpected end of pickle data (BINGET)"));
                }
                let idx = u32::from(data[pos]);
                pos += 1;
                let val = memo
                    .get(&idx)
                    .ok_or_else(|| pickle_err(&format!("BINGET: memo index {idx} not found")))?
                    .clone();
                stack.push(val);
            }

            LONG_BINGET => {
                if pos + 4 > data.len() {
                    return Err(pickle_err("unexpected end of pickle data (LONG_BINGET)"));
                }
                let idx =
                    u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
                pos += 4;
                let val = memo
                    .get(&idx)
                    .ok_or_else(|| pickle_err(&format!("LONG_BINGET: memo index {idx} not found")))?
                    .clone();
                stack.push(val);
            }

            GLOBAL => {
                let module = read_line(data, &mut pos)?;
                let name = read_line(data, &mut pos)?;
                stack.push(PickleValue::Global { module, name });
            }

            STACK_GLOBAL => {
                let name_val = stack
                    .pop()
                    .ok_or_else(|| pickle_err("STACK_GLOBAL underflow"))?;
                let module_val = stack
                    .pop()
                    .ok_or_else(|| pickle_err("STACK_GLOBAL underflow"))?;
                let PickleValue::String(module) = module_val else {
                    return Err(pickle_err("STACK_GLOBAL: module is not a string"));
                };
                let PickleValue::String(name) = name_val else {
                    return Err(pickle_err("STACK_GLOBAL: name is not a string"));
                };
                stack.push(PickleValue::Global { module, name });
            }

            SHORT_BINUNICODE => {
                if pos >= data.len() {
                    return Err(pickle_err("unexpected end (SHORT_BINUNICODE len)"));
                }
                let len = data[pos] as usize;
                pos += 1;
                if pos + len > data.len() {
                    return Err(pickle_err("unexpected end (SHORT_BINUNICODE data)"));
                }
                let s = std::str::from_utf8(&data[pos..pos + len])
                    .map_err(|_| pickle_err("SHORT_BINUNICODE: invalid UTF-8"))?;
                pos += len;
                stack.push(PickleValue::String(s.to_string()));
            }

            BINUNICODE => {
                if pos + 4 > data.len() {
                    return Err(pickle_err("unexpected end (BINUNICODE len)"));
                }
                let len =
                    u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                        as usize;
                pos += 4;
                if pos + len > data.len() {
                    return Err(pickle_err("unexpected end (BINUNICODE data)"));
                }
                let s = std::str::from_utf8(&data[pos..pos + len])
                    .map_err(|_| pickle_err("BINUNICODE: invalid UTF-8"))?;
                pos += len;
                stack.push(PickleValue::String(s.to_string()));
            }

            SHORT_BINSTRING => {
                if pos >= data.len() {
                    return Err(pickle_err("unexpected end (SHORT_BINSTRING len)"));
                }
                let len = data[pos] as usize;
                pos += 1;
                if pos + len > data.len() {
                    return Err(pickle_err("unexpected end (SHORT_BINSTRING data)"));
                }
                let s = String::from_utf8_lossy(&data[pos..pos + len]).into_owned();
                pos += len;
                stack.push(PickleValue::String(s));
            }

            BINSTRING => {
                if pos + 4 > data.len() {
                    return Err(pickle_err("unexpected end (BINSTRING len)"));
                }
                let len =
                    i32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                        as usize;
                pos += 4;
                if pos + len > data.len() {
                    return Err(pickle_err("unexpected end (BINSTRING data)"));
                }
                let s = String::from_utf8_lossy(&data[pos..pos + len]).into_owned();
                pos += len;
                stack.push(PickleValue::String(s));
            }

            SHORT_BINBYTES => {
                if pos >= data.len() {
                    return Err(pickle_err("unexpected end (SHORT_BINBYTES len)"));
                }
                let len = data[pos] as usize;
                pos += 1;
                if pos + len > data.len() {
                    return Err(pickle_err("unexpected end (SHORT_BINBYTES data)"));
                }
                let bytes = data[pos..pos + len].to_vec();
                pos += len;
                stack.push(PickleValue::Bytes(bytes));
            }

            BINBYTES => {
                if pos + 4 > data.len() {
                    return Err(pickle_err("unexpected end (BINBYTES len)"));
                }
                let len =
                    u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                        as usize;
                pos += 4;
                if pos + len > data.len() {
                    return Err(pickle_err("unexpected end (BINBYTES data)"));
                }
                let bytes = data[pos..pos + len].to_vec();
                pos += len;
                stack.push(PickleValue::Bytes(bytes));
            }

            BININT1 => {
                if pos >= data.len() {
                    return Err(pickle_err("unexpected end (BININT1)"));
                }
                let v = data[pos];
                pos += 1;
                stack.push(PickleValue::Int(i64::from(v)));
            }

            BININT2 => {
                if pos + 2 > data.len() {
                    return Err(pickle_err("unexpected end (BININT2)"));
                }
                let v = u16::from_le_bytes([data[pos], data[pos + 1]]);
                pos += 2;
                stack.push(PickleValue::Int(i64::from(v)));
            }

            BININT => {
                if pos + 4 > data.len() {
                    return Err(pickle_err("unexpected end (BININT)"));
                }
                let v =
                    i32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
                pos += 4;
                stack.push(PickleValue::Int(i64::from(v)));
            }

            BINFLOAT => {
                if pos + 8 > data.len() {
                    return Err(pickle_err("unexpected end (BINFLOAT)"));
                }
                let v = f64::from_be_bytes([
                    data[pos],
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                    data[pos + 4],
                    data[pos + 5],
                    data[pos + 6],
                    data[pos + 7],
                ]);
                pos += 8;
                stack.push(PickleValue::Float(v));
            }

            OP_NONE => {
                stack.push(PickleValue::None);
            }

            NEWTRUE => {
                stack.push(PickleValue::Bool(true));
            }

            NEWFALSE => {
                stack.push(PickleValue::Bool(false));
            }

            REDUCE => {
                let args = stack.pop().ok_or_else(|| pickle_err("REDUCE underflow"))?;
                let callable = stack.pop().ok_or_else(|| pickle_err("REDUCE underflow"))?;
                stack.push(PickleValue::Reduce {
                    callable: Box::new(callable),
                    args: Box::new(args),
                });
            }

            NEWOBJ => {
                // NEWOBJ: pop args, pop cls, push cls.__new__(cls, *args).
                // We treat this identically to REDUCE for interpretation.
                let args = stack.pop().ok_or_else(|| pickle_err("NEWOBJ underflow"))?;
                let cls = stack.pop().ok_or_else(|| pickle_err("NEWOBJ underflow"))?;
                stack.push(PickleValue::Reduce {
                    callable: Box::new(cls),
                    args: Box::new(args),
                });
            }

            BUILD => {
                let state = stack.pop().ok_or_else(|| pickle_err("BUILD underflow"))?;
                let obj = stack.pop().ok_or_else(|| pickle_err("BUILD underflow"))?;
                stack.push(PickleValue::Build {
                    obj: Box::new(obj),
                    state: Box::new(state),
                });
            }

            SETITEM => {
                let value = stack.pop().ok_or_else(|| pickle_err("SETITEM underflow"))?;
                let key = stack.pop().ok_or_else(|| pickle_err("SETITEM underflow"))?;
                match stack.last_mut() {
                    Some(PickleValue::Dict(entries)) => {
                        entries.push((key, value));
                    }
                    _ => return Err(pickle_err("SETITEM: TOS is not a dict")),
                }
            }

            SETITEMS => {
                let items = pop_to_mark(&mut stack, &mut mark_stack)?;
                if items.len() % 2 != 0 {
                    return Err(pickle_err("SETITEMS: odd number of items"));
                }
                match stack.last_mut() {
                    Some(PickleValue::Dict(entries)) => {
                        for pair in items.chunks_exact(2) {
                            entries.push((pair[0].clone(), pair[1].clone()));
                        }
                    }
                    _ => return Err(pickle_err("SETITEMS: TOS is not a dict")),
                }
            }

            APPEND => {
                let item = stack.pop().ok_or_else(|| pickle_err("APPEND underflow"))?;
                match stack.last_mut() {
                    Some(PickleValue::List(items)) => {
                        items.push(item);
                    }
                    _ => return Err(pickle_err("APPEND: TOS is not a list")),
                }
            }

            APPENDS => {
                let items = pop_to_mark(&mut stack, &mut mark_stack)?;
                match stack.last_mut() {
                    Some(PickleValue::List(items_ref)) => {
                        items_ref.extend(items);
                    }
                    _ => return Err(pickle_err("APPENDS: TOS is not a list")),
                }
            }

            BINPERSID => {
                let pid = stack
                    .pop()
                    .ok_or_else(|| pickle_err("BINPERSID underflow"))?;
                stack.push(PickleValue::PersistentLoad(Box::new(pid)));
            }

            STOP => {
                break;
            }

            other => {
                return Err(pickle_err(&format!(
                    "unsupported pickle opcode 0x{:02x} at offset {}",
                    other,
                    pos - 1
                )));
            }
        }
    }

    stack
        .pop()
        .ok_or_else(|| pickle_err("pickle stack empty at STOP"))
}

/// Read a newline-terminated ASCII string from pickle data.
fn read_line(data: &[u8], pos: &mut usize) -> FerrotorchResult<String> {
    let start = *pos;
    while *pos < data.len() && data[*pos] != b'\n' {
        *pos += 1;
    }
    if *pos >= data.len() {
        return Err(pickle_err("unterminated line in pickle"));
    }
    let line = &data[start..*pos];
    *pos += 1; // skip newline
    std::str::from_utf8(line)
        .map(std::string::ToString::to_string)
        .map_err(|_| pickle_err("non-UTF-8 line in pickle"))
}

/// Pop items from the stack back to the most recent mark.
fn pop_to_mark(
    stack: &mut Vec<PickleValue>,
    mark_stack: &mut Vec<usize>,
) -> FerrotorchResult<Vec<PickleValue>> {
    let mark_pos = mark_stack
        .pop()
        .ok_or_else(|| pickle_err("MARK stack underflow"))?;
    if mark_pos > stack.len() {
        return Err(pickle_err("MARK position beyond stack"));
    }
    Ok(stack.split_off(mark_pos))
}

fn pickle_err(msg: &str) -> FerrotorchError {
    FerrotorchError::InvalidArgument {
        message: format!("pytorch pickle: {msg}"),
    }
}

// ---------------------------------------------------------------------------
// Interpretation: walk PickleValue tree -> tensor metadata
// ---------------------------------------------------------------------------

/// Metadata extracted from a single tensor's pickle representation.
#[derive(Debug, Clone)]
struct TensorInfo {
    /// Parameter name (e.g., `"layer1.weight"`).
    name: String,
    /// Shape as a list of dimension sizes.
    shape: Vec<usize>,
    /// The storage key (`"0"`, `"1"`, ...) referencing `archive/data/{key}`.
    storage_key: String,
    /// Byte offset within the storage blob (usually 0).
    storage_offset: usize,
    /// `PyTorch` dtype string (`"Float"`, `"Double"`, `"Half"`, `"BFloat16"`).
    dtype_str: String,
}

/// Extract the state dict entries from the parsed pickle tree.
///
/// `PyTorch` state dicts are serialized as:
/// ```text
/// GLOBAL 'collections' 'OrderedDict'
/// REDUCE (EMPTY_TUPLE)
/// BUILD (list of (key, tensor) pairs)
/// ```
///
/// Each tensor value is:
/// ```text
/// GLOBAL 'torch._utils' '_rebuild_tensor_v2'
/// REDUCE (PERSISTENT_LOAD(..), offset, shape, strides)
/// ```
fn extract_state_dict(root: &PickleValue) -> FerrotorchResult<Vec<TensorInfo>> {
    let entries = find_dict_entries(root)?;
    let mut tensors = Vec::new();

    for (key_val, val) in &entries {
        let name = match key_val {
            PickleValue::String(s) => s.clone(),
            _ => continue,
        };

        if let Some(info) = try_extract_tensor_info(&name, val) {
            tensors.push(info);
        }
    }

    Ok(tensors)
}

/// Navigate the pickle tree to find the dict/OrderedDict entries.
fn find_dict_entries(value: &PickleValue) -> FerrotorchResult<Vec<(PickleValue, PickleValue)>> {
    match value {
        PickleValue::Dict(entries) => Ok(entries.clone()),

        // OrderedDict pattern:
        // Build { obj: Reduce { callable: Global("collections", "OrderedDict"), .. }, state: List[..] }
        PickleValue::Build { obj, state } => match state.as_ref() {
            PickleValue::List(items) => {
                let mut entries = Vec::new();
                for item in items {
                    if let PickleValue::Tuple(pair) = item {
                        if pair.len() == 2 {
                            entries.push((pair[0].clone(), pair[1].clone()));
                        }
                    }
                }
                if !entries.is_empty() {
                    return Ok(entries);
                }
                find_dict_entries(obj)
            }
            PickleValue::Dict(entries) => Ok(entries.clone()),
            PickleValue::Tuple(items) if items.len() == 1 => {
                if let PickleValue::List(inner) = &items[0] {
                    let mut entries = Vec::new();
                    for item in inner {
                        if let PickleValue::Tuple(pair) = item {
                            if pair.len() == 2 {
                                entries.push((pair[0].clone(), pair[1].clone()));
                            }
                        }
                    }
                    return Ok(entries);
                }
                find_dict_entries(obj)
            }
            _ => find_dict_entries(obj),
        },

        PickleValue::Reduce { callable: _, args } => find_dict_entries(args),

        PickleValue::Tuple(items) => {
            for item in items {
                if let Ok(entries) = find_dict_entries(item) {
                    if !entries.is_empty() {
                        return Ok(entries);
                    }
                }
            }
            Ok(Vec::new())
        }

        _ => Ok(Vec::new()),
    }
}

/// Try to extract tensor info from a pickle value that represents a tensor rebuild.
fn try_extract_tensor_info(name: &str, value: &PickleValue) -> Option<TensorInfo> {
    let reduce = match value {
        PickleValue::Reduce { callable, args } => Some((callable.as_ref(), args.as_ref())),
        PickleValue::Build { obj, .. } => match obj.as_ref() {
            PickleValue::Reduce { callable, args } => Some((callable.as_ref(), args.as_ref())),
            _ => None,
        },
        _ => None,
    };

    let (callable, args) = reduce?;

    // Verify this is a tensor rebuild call.
    let is_rebuild = match callable {
        PickleValue::Global { module, name } => {
            (module == "torch._utils"
                && (name == "_rebuild_tensor_v2" || name == "_rebuild_tensor_v3"))
                || (module == "torch" && name == "_utils._rebuild_tensor_v2")
        }
        _ => false,
    };

    if !is_rebuild {
        return None;
    }

    let PickleValue::Tuple(args_vec) = args else {
        return None;
    };

    if args_vec.len() < 4 {
        return None;
    }

    let persistent_load = &args_vec[0];
    let (storage_key, dtype_str) = extract_storage_info(persistent_load)?;

    let storage_offset = match &args_vec[1] {
        PickleValue::Int(n) => *n as usize,
        _ => 0,
    };

    let shape = match &args_vec[2] {
        PickleValue::Tuple(dims) => dims
            .iter()
            .filter_map(|d| match d {
                PickleValue::Int(n) => Some(*n as usize),
                _ => None,
            })
            .collect(),
        _ => return None,
    };

    Some(TensorInfo {
        name: name.to_string(),
        shape,
        storage_key,
        storage_offset,
        dtype_str,
    })
}

/// Extract storage key and dtype from a `PersistentLoad` reference.
///
/// The persistent load tuple is: `('storage', storage_type, key, device, numel)`
fn extract_storage_info(value: &PickleValue) -> Option<(String, String)> {
    let tuple = match value {
        PickleValue::PersistentLoad(inner) => match inner.as_ref() {
            PickleValue::Tuple(v) => v,
            _ => return None,
        },
        PickleValue::Tuple(v) => v,
        _ => return None,
    };

    if tuple.len() < 3 {
        return None;
    }

    let key = match &tuple[2] {
        PickleValue::String(s) => s.clone(),
        PickleValue::Int(n) => n.to_string(),
        _ => return None,
    };

    let dtype_str = match &tuple[1] {
        PickleValue::Global { name, .. } => storage_type_to_dtype(name),
        PickleValue::Reduce { callable, .. } => match callable.as_ref() {
            PickleValue::Global { name, .. } => storage_type_to_dtype(name),
            _ => "Float".to_string(),
        },
        _ => "Float".to_string(),
    };

    Some((key, dtype_str))
}

/// Map a `PyTorch` storage class name to a dtype string.
fn storage_type_to_dtype(name: &str) -> String {
    if name.contains("Double") {
        "Double".to_string()
    } else if name.contains("BFloat") {
        "BFloat16".to_string()
    } else if name.contains("Half") {
        "Half".to_string()
    } else if name.contains("Float") {
        "Float".to_string()
    } else if name.contains("Long") {
        "Long".to_string()
    } else if name.contains("Int") {
        "Int".to_string()
    } else if name.contains("Byte") || name.contains("Uint8") {
        "Byte".to_string()
    } else {
        "Float".to_string()
    }
}

/// Number of bytes per element for a given `PyTorch` dtype string.
// Several PyTorch dtypes share an element width (e.g. `Float` and `Int` are
// both 4 bytes); merging them via `|` would obscure the spec mapping that
// callers cross-reference against PyTorch's `torch.dtype.itemsize`. Keep
// the one-arm-per-dtype shape and silence the lint.
#[allow(clippy::match_same_arms)]
fn dtype_element_size(dtype: &str) -> usize {
    match dtype {
        "Float" => 4,
        "Double" => 8,
        "Half" => 2,
        "BFloat16" => 2,
        "Int" => 4,
        "Long" => 8,
        "Byte" => 1,
        _ => 4,
    }
}

// ---------------------------------------------------------------------------
// ZIP reading + tensor loading
// ---------------------------------------------------------------------------

/// Load a `PyTorch` `.pt` / `.pth` state dict file and return it as a
/// `StateDict<T>`.
///
/// The function:
/// 1. Opens the file as a ZIP archive.
/// 2. Locates `data.pkl` (or `archive/data.pkl`) and parses the pickle.
/// 3. Walks the pickle tree to find tensor rebuild entries.
/// 4. Reads raw bytes from `archive/data/{key}` for each tensor.
/// 5. Converts bytes to `Tensor<T>`, promoting f16 -> f32 if `T = f32`.
///
/// # Type parameter
///
/// `T` is the output element type (`f32` or `f64`). Tensors stored as a
/// different dtype will be converted:
/// - `f32` stored, `T = f32`: no conversion.
/// - `f64` stored, `T = f64`: no conversion.
/// - `f16` stored, `T = f32`: promoted to f32.
/// - Other mismatches are handled by promotion/demotion.
///
/// # Errors
///
/// Returns an error if the file is not a valid ZIP, the pickle cannot be
/// parsed, or a tensor dtype cannot be converted to `T`.
pub fn load_pytorch_state_dict<T: Float>(path: impl AsRef<Path>) -> FerrotorchResult<StateDict<T>> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to open pytorch file {}: {e}", path.display()),
    })?;

    let archive = zip::ZipArchive::new(file).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to read ZIP archive {}: {e}", path.display()),
    })?;

    load_pytorch_state_dict_inner(archive)
}

/// Memory-mapped variant of [`load_pytorch_state_dict`] (#629). Same return
/// contract, but uses `mmap2::Mmap` + a `Cursor<&[u8]>` instead of an open
/// `File` for the underlying ZIP reader. The mmap is dropped before this
/// function returns; tensor data is copied into owned `Tensor<T>` buffers,
/// so callers don't inherit any file-lifetime invariants.
///
/// The win is the same as [`crate::load_safetensors_mmap`] / `load_gguf_mmap`:
/// the OS page cache holds raw archive bytes instead of reading them into a
/// heap `Vec<u8>` up front. The pickle parser still allocates internally
/// while it walks the bytecode, but the outer ZIP reader is now lazy.
pub fn load_pytorch_state_dict_mmap<T: Float>(
    path: impl AsRef<Path>,
) -> FerrotorchResult<StateDict<T>> {
    let path = path.as_ref();
    let file = std::fs::File::open(path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to open pytorch file {}: {e}", path.display()),
    })?;

    // SAFETY: the mmap is dropped before this function returns. All
    // tensor data is copied into owned `Tensor<T>` buffers via
    // `convert_bytes_to_float`. The file must not be mutated while the
    // mmap is live, matching the safetensors/GGUF mmap contracts.
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to mmap pytorch file {}: {e}", path.display()),
    })?;

    let cursor = Cursor::new(&mmap[..]);
    let archive = zip::ZipArchive::new(cursor).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to read ZIP archive {}: {e}", path.display()),
    })?;

    load_pytorch_state_dict_inner(archive)
}

/// Shared inner: takes any `ZipArchive<R>` and produces a `StateDict<T>`.
/// Both [`load_pytorch_state_dict`] (file-backed) and
/// [`load_pytorch_state_dict_mmap`] (mmap-backed) funnel here. (#629)
fn load_pytorch_state_dict_inner<T: Float, R: Read + std::io::Seek>(
    mut archive: zip::ZipArchive<R>,
) -> FerrotorchResult<StateDict<T>> {
    // Find the pickle file.
    let pkl_name = find_pkl_name(&mut archive)?;

    // Read pickle bytes.
    let pkl_bytes = read_zip_entry(&mut archive, &pkl_name)?;

    // Parse pickle.
    let root = parse_pickle(&pkl_bytes)?;

    // Extract tensor info from the pickle tree.
    let tensor_infos = extract_state_dict(&root)?;

    if tensor_infos.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "no tensors found in pytorch state dict".into(),
        });
    }

    // Determine the archive prefix (e.g., "archive/data/" or just "data/").
    let data_prefix = find_data_prefix(&mut archive, &tensor_infos);

    // Load each tensor.
    let mut state: StateDict<T> = HashMap::with_capacity(tensor_infos.len());

    for info in &tensor_infos {
        let data_path = format!("{}{}", data_prefix, info.storage_key);
        let raw_bytes = read_zip_entry(&mut archive, &data_path)?;

        let src_elem_size = dtype_element_size(&info.dtype_str);
        let numel: usize = if info.shape.is_empty() {
            1
        } else {
            info.shape.iter().product()
        };

        let byte_offset = info.storage_offset * src_elem_size;
        let byte_length = numel * src_elem_size;

        if byte_offset + byte_length > raw_bytes.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "tensor \"{}\" requires bytes [{}..{}] but storage blob has {} bytes",
                    info.name,
                    byte_offset,
                    byte_offset + byte_length,
                    raw_bytes.len()
                ),
            });
        }

        let src_bytes = &raw_bytes[byte_offset..byte_offset + byte_length];

        let data = convert_bytes_to_float::<T>(src_bytes, &info.dtype_str, numel)?;

        let storage = TensorStorage::cpu(data);
        let tensor = Tensor::from_storage(storage, info.shape.clone(), false)?;
        state.insert(info.name.clone(), tensor);
    }

    Ok(state)
}

/// Find the pickle file name inside the ZIP archive.
fn find_pkl_name<R: Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
) -> FerrotorchResult<String> {
    let names: Vec<String> = (0..archive.len())
        .filter_map(|i| archive.by_index(i).ok().map(|f| f.name().to_string()))
        .collect();

    for candidate in &["archive/data.pkl", "data.pkl"] {
        if names.iter().any(|n| n == *candidate) {
            return Ok(candidate.to_string());
        }
    }

    // Try any file ending in .pkl (case-insensitive — PyTorch's archive
    // format uses lowercase, but some downstream tooling capitalises).
    for name in &names {
        if std::path::Path::new(name)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("pkl"))
        {
            return Ok(name.clone());
        }
    }

    Err(FerrotorchError::InvalidArgument {
        message: format!("no .pkl file found in pytorch archive. Files: {names:?}"),
    })
}

/// Determine the prefix path for data blobs.
fn find_data_prefix<R: Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    infos: &[TensorInfo],
) -> String {
    if infos.is_empty() {
        return "archive/data/".to_string();
    }

    let first_key = &infos[0].storage_key;

    let names: Vec<String> = (0..archive.len())
        .filter_map(|i| archive.by_index(i).ok().map(|f| f.name().to_string()))
        .collect();

    for prefix in &["archive/data/", "data/", ""] {
        let candidate = format!("{prefix}{first_key}");
        if names.iter().any(|n| n == &candidate) {
            return prefix.to_string();
        }
    }

    "archive/data/".to_string()
}

/// Read a named entry from the ZIP archive into a byte vector.
fn read_zip_entry<R: Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    name: &str,
) -> FerrotorchResult<Vec<u8>> {
    let mut entry = archive
        .by_name(name)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("ZIP entry \"{name}\" not found: {e}"),
        })?;

    let mut buf = Vec::with_capacity(entry.size() as usize);
    entry
        .read_to_end(&mut buf)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to read ZIP entry \"{name}\": {e}"),
        })?;

    Ok(buf)
}

/// Convert raw little-endian bytes to `Vec<T>`, handling dtype promotion.
fn convert_bytes_to_float<T: Float>(
    bytes: &[u8],
    dtype: &str,
    numel: usize,
) -> FerrotorchResult<Vec<T>> {
    let target_size = std::mem::size_of::<T>();

    match dtype {
        "Float" => {
            if target_size == 4 {
                // T == f32: byte-for-byte reinterpretation is correct.
                reinterpret_le_bytes::<T>(bytes, 4)
            } else if target_size == 8 {
                // T == f64: f32 -> f64 promotion via safe `cast`.
                let f32s = reinterpret_le_bytes::<f32>(bytes, 4)?;
                f32s.into_iter().map(cast::<f32, T>).collect()
            } else {
                // T == bf16 (size 2) or another exotic Float impl.
                let f32s = reinterpret_le_bytes::<f32>(bytes, 4)?;
                f32s.into_iter().map(cast::<f32, T>).collect()
            }
        }

        "Double" => {
            if target_size == 8 {
                // T == f64: byte-for-byte reinterpretation is correct.
                reinterpret_le_bytes::<T>(bytes, 8)
            } else if target_size == 4 {
                // T == f32: f64 -> f32 demotion via safe `cast`.
                let f64s = reinterpret_le_bytes::<f64>(bytes, 8)?;
                f64s.into_iter().map(cast::<f64, T>).collect()
            } else {
                // T == bf16 or other narrow Float; demote through f64.
                let f64s = reinterpret_le_bytes::<f64>(bytes, 8)?;
                f64s.into_iter().map(cast::<f64, T>).collect()
            }
        }

        "Half" => {
            if numel * 2 > bytes.len() {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "Half tensor requires {} bytes but only {} available",
                        numel * 2,
                        bytes.len()
                    ),
                });
            }

            bytes
                .chunks_exact(2)
                .take(numel)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let f32_val = f16_to_f32(bits);
                    cast::<f32, T>(f32_val)
                })
                .collect()
        }

        "BFloat16" => {
            if numel * 2 > bytes.len() {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "BFloat16 tensor requires {} bytes but only {} available",
                        numel * 2,
                        bytes.len()
                    ),
                });
            }

            bytes
                .chunks_exact(2)
                .take(numel)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let f32_val = bf16_to_f32(bits);
                    cast::<f32, T>(f32_val)
                })
                .collect()
        }

        other => Err(FerrotorchError::DtypeMismatch {
            expected: "Float, Double, Half, or BFloat16".to_string(),
            got: other.to_string(),
        }),
    }
}

/// Reinterpret raw little-endian bytes as a `Vec<T>`.
///
/// `elem_size` is the on-disk byte width of one `T` element (4 for `f32`, 8
/// for `f64`). The function returns an error if `elem_size != size_of::<T>()`
/// rather than producing UB by reading more or fewer bytes than `T` occupies.
///
/// # Errors
///
/// Returns `FerrotorchError::InvalidArgument` if `elem_size` disagrees with
/// `size_of::<T>()`. This guards every `unsafe` site below from a size
/// mismatch that would corrupt memory.
fn reinterpret_le_bytes<T: Copy>(bytes: &[u8], elem_size: usize) -> FerrotorchResult<Vec<T>> {
    if elem_size != std::mem::size_of::<T>() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "reinterpret_le_bytes: elem_size {} does not match size_of::<{}>() = {}",
                elem_size,
                std::any::type_name::<T>(),
                std::mem::size_of::<T>(),
            ),
        });
    }
    Ok(bytes
        .chunks_exact(elem_size)
        .map(|chunk| {
            let mut buf = [0u8; 8];
            buf[..elem_size].copy_from_slice(chunk);
            // SAFETY: `T: Copy` so reading produces a valid, owned bit
            // pattern with no `Drop` semantics. The size precondition above
            // guarantees `elem_size == size_of::<T>()`, so `chunks_exact`
            // hands us a `chunk` of exactly `size_of::<T>()` bytes which we
            // copy fully into `buf`. `buf.as_ptr()` is valid for
            // `size_of::<T>()` bytes (the array is 8 bytes; we only read
            // `size_of::<T>()` of them, and `T` is one of f32/f64 here, both
            // <= 8). `read_unaligned` requires no alignment, so the cast
            // from `*const u8` to `*const T` carrying any address is sound.
            // Every bit pattern is valid for `f32`/`f64` (NaNs included), so
            // the read cannot produce an invalid value.
            unsafe { std::ptr::read_unaligned(buf.as_ptr().cast::<T>()) }
        })
        .collect())
}

/// Convert IEEE 754 half-precision (f16) bits to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 1);
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x3ff);

    if exponent == 0 {
        if mantissa == 0 {
            // Signed zero.
            f32::from_bits(sign << 31)
        } else {
            // Subnormal f16 -> normal f32.
            let mut m = mantissa;
            let mut e: i32 = -14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let f32_exp = ((e + 127) as u32) & 0xff;
            let f32_mantissa = m << 13;
            f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mantissa)
        }
    } else if exponent == 31 {
        if mantissa == 0 {
            f32::from_bits((sign << 31) | (0xff << 23))
        } else {
            // NaN -- preserve payload.
            f32::from_bits((sign << 31) | (0xff << 23) | (mantissa << 13))
        }
    } else {
        // Normal number.
        let f32_exp = (exponent as i32 - 15 + 127) as u32;
        let f32_mantissa = mantissa << 13;
        f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mantissa)
    }
}

/// Convert `BFloat16` bits to f32.
///
/// `BFloat16` is the upper 16 bits of an f32, so conversion is a left-shift.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // -- Pickle parser tests --

    #[test]
    fn test_pickle_empty_dict() {
        let data = [0x80, 0x02, EMPTY_DICT, STOP];
        let val = parse_pickle(&data).unwrap();
        match val {
            PickleValue::Dict(entries) => assert!(entries.is_empty()),
            other => panic!("expected Dict, got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_empty_list() {
        let data = [0x80, 0x02, EMPTY_LIST, STOP];
        let val = parse_pickle(&data).unwrap();
        match val {
            PickleValue::List(items) => assert!(items.is_empty()),
            other => panic!("expected List, got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_empty_tuple() {
        let data = [0x80, 0x02, EMPTY_TUPLE, STOP];
        let val = parse_pickle(&data).unwrap();
        match val {
            PickleValue::Tuple(items) => assert!(items.is_empty()),
            other => panic!("expected Tuple, got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_none() {
        let data = [0x80, 0x02, OP_NONE, STOP];
        let val = parse_pickle(&data).unwrap();
        assert!(matches!(val, PickleValue::None));
    }

    #[test]
    fn test_pickle_true_false() {
        let data_t = [0x80, 0x02, NEWTRUE, STOP];
        match parse_pickle(&data_t).unwrap() {
            PickleValue::Bool(true) => {}
            other => panic!("expected Bool(true), got: {other:?}"),
        }

        let data_f = [0x80, 0x02, NEWFALSE, STOP];
        match parse_pickle(&data_f).unwrap() {
            PickleValue::Bool(false) => {}
            other => panic!("expected Bool(false), got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_binint1() {
        let data = [0x80, 0x02, BININT1, 42, STOP];
        match parse_pickle(&data).unwrap() {
            PickleValue::Int(v) => assert_eq!(v, 42),
            other => panic!("expected Int(42), got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_binint2() {
        let data = [0x80, 0x02, BININT2, 0x00, 0x01, STOP];
        match parse_pickle(&data).unwrap() {
            PickleValue::Int(v) => assert_eq!(v, 256),
            other => panic!("expected Int(256), got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_binint() {
        let data = [0x80, 0x02, BININT, 0xff, 0xff, 0xff, 0xff, STOP];
        match parse_pickle(&data).unwrap() {
            PickleValue::Int(v) => assert_eq!(v, -1),
            other => panic!("expected Int(-1), got: {other:?}"),
        }
    }

    #[test]
    #[allow(clippy::approx_constant)] // 3.14 is an arbitrary pickle round-trip value, not π.
    fn test_pickle_binfloat() {
        let f_bytes = 3.14f64.to_be_bytes();
        let mut data = vec![0x80, 0x02, BINFLOAT];
        data.extend_from_slice(&f_bytes);
        data.push(STOP);
        match parse_pickle(&data).unwrap() {
            PickleValue::Float(v) => assert!((v - 3.14).abs() < 1e-12),
            other => panic!("expected Float(3.14), got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_short_binunicode() {
        let mut data = vec![0x80, 0x02, SHORT_BINUNICODE, 0x05];
        data.extend_from_slice(b"hello");
        data.push(STOP);
        match parse_pickle(&data).unwrap() {
            PickleValue::String(s) => assert_eq!(s, "hello"),
            other => panic!("expected String, got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_binunicode() {
        let mut data = vec![0x80, 0x02, BINUNICODE, 0x03, 0x00, 0x00, 0x00];
        data.extend_from_slice(b"abc");
        data.push(STOP);
        match parse_pickle(&data).unwrap() {
            PickleValue::String(s) => assert_eq!(s, "abc"),
            other => panic!("expected String, got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_tuple1() {
        let data = [0x80, 0x02, BININT1, 7, TUPLE1, STOP];
        match parse_pickle(&data).unwrap() {
            PickleValue::Tuple(v) => {
                assert_eq!(v.len(), 1);
                assert!(matches!(&v[0], PickleValue::Int(7)));
            }
            other => panic!("expected Tuple, got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_tuple2() {
        let data = [0x80, 0x02, BININT1, 1, BININT1, 2, TUPLE2, STOP];
        match parse_pickle(&data).unwrap() {
            PickleValue::Tuple(v) => {
                assert_eq!(v.len(), 2);
                assert!(matches!(&v[0], PickleValue::Int(1)));
                assert!(matches!(&v[1], PickleValue::Int(2)));
            }
            other => panic!("expected Tuple, got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_tuple3() {
        let data = [0x80, 0x02, BININT1, 1, BININT1, 2, BININT1, 3, TUPLE3, STOP];
        match parse_pickle(&data).unwrap() {
            PickleValue::Tuple(v) => assert_eq!(v.len(), 3),
            other => panic!("expected Tuple, got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_mark_tuple() {
        let data = [0x80, 0x02, MARK, BININT1, 10, BININT1, 20, TUPLE, STOP];
        match parse_pickle(&data).unwrap() {
            PickleValue::Tuple(v) => assert_eq!(v.len(), 2),
            other => panic!("expected Tuple, got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_memo_put_get() {
        // SHORT_BINUNICODE "x", BINPUT 0, BINGET 0, TUPLE2
        let mut data = vec![0x80, 0x02];
        data.extend_from_slice(&[SHORT_BINUNICODE, 0x01, b'x']);
        data.push(BINPUT);
        data.push(0x00);
        data.push(BINGET);
        data.push(0x00);
        data.push(TUPLE2);
        data.push(STOP);
        match parse_pickle(&data).unwrap() {
            PickleValue::Tuple(v) => {
                assert_eq!(v.len(), 2);
                assert!(matches!(&v[0], PickleValue::String(s) if s == "x"));
                assert!(matches!(&v[1], PickleValue::String(s) if s == "x"));
            }
            other => panic!("expected Tuple, got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_global() {
        let mut data = vec![0x80, 0x02, GLOBAL];
        data.extend_from_slice(b"collections\nOrderedDict\n");
        data.push(STOP);
        match parse_pickle(&data).unwrap() {
            PickleValue::Global { module, name } => {
                assert_eq!(module, "collections");
                assert_eq!(name, "OrderedDict");
            }
            other => panic!("expected Global, got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_reduce() {
        let mut data = vec![0x80, 0x02, GLOBAL];
        data.extend_from_slice(b"collections\nOrderedDict\n");
        data.push(EMPTY_TUPLE);
        data.push(REDUCE);
        data.push(STOP);
        match parse_pickle(&data).unwrap() {
            PickleValue::Reduce { callable, args } => {
                assert!(matches!(
                    *callable,
                    PickleValue::Global { ref module, ref name }
                        if module == "collections" && name == "OrderedDict"
                ));
                assert!(matches!(*args, PickleValue::Tuple(ref v) if v.is_empty()));
            }
            other => panic!("expected Reduce, got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_dict_setitem() {
        let mut data = vec![0x80, 0x02, EMPTY_DICT];
        data.extend_from_slice(&[SHORT_BINUNICODE, 0x01, b'a']);
        data.push(BININT1);
        data.push(1);
        data.push(SETITEM);
        data.push(STOP);
        match parse_pickle(&data).unwrap() {
            PickleValue::Dict(entries) => {
                assert_eq!(entries.len(), 1);
                assert!(matches!(&entries[0].0, PickleValue::String(s) if s == "a"));
                assert!(matches!(&entries[0].1, PickleValue::Int(1)));
            }
            other => panic!("expected Dict, got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_dict_setitems() {
        let mut data = vec![0x80, 0x02, EMPTY_DICT, MARK];
        data.extend_from_slice(&[SHORT_BINUNICODE, 0x01, b'a']);
        data.extend_from_slice(&[BININT1, 1]);
        data.extend_from_slice(&[SHORT_BINUNICODE, 0x01, b'b']);
        data.extend_from_slice(&[BININT1, 2]);
        data.push(SETITEMS);
        data.push(STOP);
        match parse_pickle(&data).unwrap() {
            PickleValue::Dict(entries) => assert_eq!(entries.len(), 2),
            other => panic!("expected Dict, got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_list_append() {
        let data = [
            0x80, 0x02, EMPTY_LIST, BININT1, 1, APPEND, BININT1, 2, APPEND, STOP,
        ];
        match parse_pickle(&data).unwrap() {
            PickleValue::List(items) => assert_eq!(items.len(), 2),
            other => panic!("expected List, got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_list_appends() {
        let data = [
            0x80, 0x02, EMPTY_LIST, MARK, BININT1, 1, BININT1, 2, APPENDS, STOP,
        ];
        match parse_pickle(&data).unwrap() {
            PickleValue::List(items) => assert_eq!(items.len(), 2),
            other => panic!("expected List, got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_binpersid() {
        let data = [0x80, 0x02, BININT1, 99, BINPERSID, STOP];
        match parse_pickle(&data).unwrap() {
            PickleValue::PersistentLoad(inner) => {
                assert!(matches!(*inner, PickleValue::Int(99)));
            }
            other => panic!("expected PersistentLoad, got: {other:?}"),
        }
    }

    #[test]
    fn test_pickle_build() {
        let data = [0x80, 0x02, EMPTY_DICT, BININT1, 42, BUILD, STOP];
        match parse_pickle(&data).unwrap() {
            PickleValue::Build { obj, state } => {
                assert!(matches!(*obj, PickleValue::Dict(_)));
                assert!(matches!(*state, PickleValue::Int(42)));
            }
            other => panic!("expected Build, got: {other:?}"),
        }
    }

    // -- f16/bf16 conversion tests --

    #[test]
    fn test_f16_to_f32_normal() {
        let val = f16_to_f32(0x3C00); // 1.0
        assert!((val - 1.0).abs() < 1e-6, "got {val}");
    }

    #[test]
    // f16 zero converts to f32 zero with an exact bit pattern; strict
    // equality is the correct test (an epsilon would silently accept a
    // bug in the conversion).
    #[allow(clippy::float_cmp)]
    fn test_f16_to_f32_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert!(f16_to_f32(0x8000).is_sign_negative());
    }

    #[test]
    fn test_f16_to_f32_neg() {
        let val = f16_to_f32(0xBC00); // -1.0
        assert!((val + 1.0).abs() < 1e-6, "got {val}");
    }

    #[test]
    fn test_f16_to_f32_inf() {
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7C00) > 0.0);
        assert!(f16_to_f32(0xFC00).is_infinite());
        assert!(f16_to_f32(0xFC00) < 0.0);
    }

    #[test]
    fn test_f16_to_f32_nan() {
        assert!(f16_to_f32(0x7C01).is_nan());
    }

    #[test]
    fn test_f16_to_f32_subnormal() {
        let val = f16_to_f32(0x0001); // smallest positive subnormal
        assert!(val > 0.0);
        assert!((val - 5.960_464_5e-8).abs() < 1e-14, "got {val}");
    }

    #[test]
    fn test_bf16_to_f32() {
        let val = bf16_to_f32(0x3F80); // 1.0
        assert!((val - 1.0).abs() < 1e-6, "got {val}");

        let val2 = bf16_to_f32(0xC000); // -2.0
        assert!((val2 + 2.0).abs() < 1e-6, "got {val2}");
    }

    // -- dtype conversion tests --

    #[test]
    #[allow(clippy::approx_constant)] // -3.14 is an arbitrary round-trip value, not -π.
    fn test_convert_f32_to_f32() {
        let src: Vec<f32> = vec![1.0, 2.5, -3.14];
        let bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result: Vec<f32> = convert_bytes_to_float(&bytes, "Float", 3).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.5).abs() < 1e-6);
        assert!((result[2] + 3.14).abs() < 1e-5);
    }

    #[test]
    #[allow(clippy::approx_constant)] // -3.14 is an arbitrary round-trip value, not -π.
    fn test_convert_f64_to_f64() {
        let src: Vec<f64> = vec![1.0, 2.5, -3.14];
        let bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result: Vec<f64> = convert_bytes_to_float(&bytes, "Double", 3).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-12);
        assert!((result[1] - 2.5).abs() < 1e-12);
        assert!((result[2] + 3.14).abs() < 1e-12);
    }

    #[test]
    fn test_convert_f32_to_f64_promotion() {
        let src: Vec<f32> = vec![1.0, -2.5];
        let bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result: Vec<f64> = convert_bytes_to_float(&bytes, "Float", 2).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] + 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_convert_f64_to_f32_demotion() {
        let src: Vec<f64> = vec![1.0, -2.5];
        let bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result: Vec<f32> = convert_bytes_to_float(&bytes, "Double", 2).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] + 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_convert_f16_to_f32() {
        // 1.0 in f16 = 0x3C00, -1.0 = 0xBC00 (little-endian byte pairs)
        let bytes = vec![0x00, 0x3C, 0x00, 0xBC];
        let result: Vec<f32> = convert_bytes_to_float(&bytes, "Half", 2).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_convert_bf16_to_f32() {
        // 1.0 in bf16 = 0x3F80
        let bytes = vec![0x80, 0x3F];
        let result: Vec<f32> = convert_bytes_to_float(&bytes, "BFloat16", 1).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_convert_unsupported_dtype() {
        let bytes = vec![0; 4];
        let result: Result<Vec<f32>, _> = convert_bytes_to_float(&bytes, "Complex64", 1);
        assert!(result.is_err());
    }

    // -- ZIP / synthetic .pt file tests --

    /// Build a minimal synthetic .pt ZIP file in memory for testing.
    fn build_synthetic_pt(tensor_data: &[f32]) -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());

        {
            let mut zip = zip::ZipWriter::new(&mut buf);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);

            let pkl = build_test_pickle(tensor_data.len());

            zip.start_file("archive/data.pkl", options).unwrap();
            zip.write_all(&pkl).unwrap();

            zip.start_file("archive/data/0", options).unwrap();
            let raw_bytes: Vec<u8> = tensor_data.iter().flat_map(|v| v.to_le_bytes()).collect();
            zip.write_all(&raw_bytes).unwrap();

            zip.finish().unwrap();
        }

        buf.into_inner()
    }

    /// Build pickle bytecodes for a minimal state dict with one f32 tensor
    /// named "weight" with shape [2, 3].
    fn build_test_pickle(numel: usize) -> Vec<u8> {
        let mut pkl = Vec::new();

        // PROTO 2
        pkl.extend_from_slice(&[0x80, 0x02]);

        // GLOBAL "collections" "OrderedDict"
        pkl.push(GLOBAL);
        pkl.extend_from_slice(b"collections\nOrderedDict\n");

        // EMPTY_TUPLE, REDUCE -> OrderedDict()
        pkl.push(EMPTY_TUPLE);
        pkl.push(REDUCE);
        pkl.push(BINPUT);
        pkl.push(0);

        // BUILD with a list of (key, tensor) tuples.
        pkl.push(EMPTY_LIST);
        pkl.push(BINPUT);
        pkl.push(1);

        pkl.push(MARK); // for APPENDS

        // key: "weight"
        pkl.push(SHORT_BINUNICODE);
        pkl.push(6);
        pkl.extend_from_slice(b"weight");

        // value: _rebuild_tensor_v2(persistent_load(...), 0, (2,3), (3,1))
        pkl.push(GLOBAL);
        pkl.extend_from_slice(b"torch._utils\n_rebuild_tensor_v2\n");

        pkl.push(MARK); // start args tuple

        // persistent_load tuple: ("storage", FloatStorage, "0", "cpu", numel)
        pkl.push(MARK);
        pkl.push(SHORT_BINUNICODE);
        pkl.push(7);
        pkl.extend_from_slice(b"storage");

        pkl.push(GLOBAL);
        pkl.extend_from_slice(b"torch\nFloatStorage\n");

        pkl.push(SHORT_BINUNICODE);
        pkl.push(1);
        pkl.push(b'0');

        pkl.push(SHORT_BINUNICODE);
        pkl.push(3);
        pkl.extend_from_slice(b"cpu");

        pkl.push(BININT1);
        pkl.push(numel as u8);

        pkl.push(TUPLE); // end storage tuple
        pkl.push(BINPERSID);

        // storage_offset = 0
        pkl.push(BININT1);
        pkl.push(0);

        // shape = (2, 3)
        pkl.push(BININT1);
        pkl.push(2);
        pkl.push(BININT1);
        pkl.push(3);
        pkl.push(TUPLE2);

        // strides = (3, 1)
        pkl.push(BININT1);
        pkl.push(3);
        pkl.push(BININT1);
        pkl.push(1);
        pkl.push(TUPLE2);

        pkl.push(TUPLE); // end args
        pkl.push(REDUCE); // _rebuild_tensor_v2(*args)

        pkl.push(BINPUT);
        pkl.push(2);

        // (key, value) tuple
        pkl.push(TUPLE2);

        pkl.push(APPENDS);

        // BUILD the OrderedDict
        pkl.push(BUILD);

        pkl.push(STOP);

        pkl
    }

    #[test]
    fn test_synthetic_pt_file_f32() {
        let tensor_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let zip_bytes = build_synthetic_pt(&tensor_data);

        let dir = std::env::temp_dir().join("ferrotorch_test_pt_f32");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.pt");
        std::fs::write(&path, &zip_bytes).unwrap();

        let state: StateDict<f32> = load_pytorch_state_dict(&path).unwrap();

        assert_eq!(state.len(), 1);
        let w = &state["weight"];
        assert_eq!(w.shape(), &[2, 3]);
        let data = w.data().unwrap();
        assert_eq!(data, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_synthetic_pt_file_f32_to_f64() {
        let tensor_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let zip_bytes = build_synthetic_pt(&tensor_data);

        let dir = std::env::temp_dir().join("ferrotorch_test_pt_f32_to_f64");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.pt");
        std::fs::write(&path, &zip_bytes).unwrap();

        let state: StateDict<f64> = load_pytorch_state_dict(&path).unwrap();

        assert_eq!(state.len(), 1);
        let w = &state["weight"];
        assert_eq!(w.shape(), &[2, 3]);
        let data = w.data().unwrap();
        for (i, &v) in data.iter().enumerate() {
            let expected = (i + 1) as f64;
            assert!(
                (v - expected).abs() < 1e-6,
                "element {i}: expected {expected}, got {v}"
            );
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_synthetic_pt_file_mmap_matches_read() {
        // (#629) The mmap-backed loader should produce byte-identical
        // output to the read-backed loader on the same file.
        let tensor_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let zip_bytes = build_synthetic_pt(&tensor_data);

        let dir = std::env::temp_dir().join("ferrotorch_test_pt_mmap");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.pt");
        std::fs::write(&path, &zip_bytes).unwrap();

        let from_read: StateDict<f32> = load_pytorch_state_dict(&path).unwrap();
        let from_mmap: StateDict<f32> = load_pytorch_state_dict_mmap(&path).unwrap();

        assert_eq!(from_read.len(), from_mmap.len());
        for (k, v_read) in &from_read {
            let v_mmap = &from_mmap[k];
            assert_eq!(v_read.shape(), v_mmap.shape());
            assert_eq!(v_read.data().unwrap(), v_mmap.data().unwrap());
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_pytorch_mmap_rejects_missing_file() {
        let result = load_pytorch_state_dict_mmap::<f32>("/nonexistent/path/model.pt");
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("failed to open"));
    }

    #[test]
    fn test_load_missing_pt_file() {
        let result = load_pytorch_state_dict::<f32>("/nonexistent/path/model.pt");
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("failed to open"));
    }

    #[test]
    fn test_load_invalid_zip() {
        let dir = std::env::temp_dir().join("ferrotorch_test_pt_invalid");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("not_a_zip.pt");
        std::fs::write(&path, b"this is not a zip file").unwrap();

        let result = load_pytorch_state_dict::<f32>(&path);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("ZIP"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_storage_type_to_dtype() {
        assert_eq!(storage_type_to_dtype("FloatStorage"), "Float");
        assert_eq!(storage_type_to_dtype("DoubleStorage"), "Double");
        assert_eq!(storage_type_to_dtype("HalfStorage"), "Half");
        assert_eq!(storage_type_to_dtype("BFloat16Storage"), "BFloat16");
        assert_eq!(storage_type_to_dtype("LongStorage"), "Long");
        assert_eq!(storage_type_to_dtype("ByteStorage"), "Byte");
    }

    #[test]
    fn test_dtype_element_size_values() {
        assert_eq!(dtype_element_size("Float"), 4);
        assert_eq!(dtype_element_size("Double"), 8);
        assert_eq!(dtype_element_size("Half"), 2);
        assert_eq!(dtype_element_size("BFloat16"), 2);
        assert_eq!(dtype_element_size("Long"), 8);
        assert_eq!(dtype_element_size("Byte"), 1);
    }

    #[test]
    fn test_pickle_parser_complex_state_dict() {
        let pkl = build_test_pickle(6);
        let root = parse_pickle(&pkl).unwrap();

        let tensors = extract_state_dict(&root).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].name, "weight");
        assert_eq!(tensors[0].shape, vec![2, 3]);
        assert_eq!(tensors[0].storage_key, "0");
        assert_eq!(tensors[0].storage_offset, 0);
        assert_eq!(tensors[0].dtype_str, "Float");
    }
}
