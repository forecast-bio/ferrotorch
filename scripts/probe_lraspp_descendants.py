#!/usr/bin/env python3
"""Probe ferrotorch Lraspp module descendant tree to verify BN-buffer
loader can resolve all backbone+head BN paths. Diagnoses the BN-buffer
silent-miss failure mode (#1142 pattern).
"""
import subprocess
import sys
from pathlib import Path

# We need a Rust-side script. Instead, let's inspect named_descendants_dyn
# via a one-off Rust binary built on the fly. But simpler: write a small
# Rust example that prints the descendant tree and run it.

EXAMPLE_PATH = Path('ferrotorch-vision/examples/dump_lraspp_descendants.rs')
EXAMPLE_PATH.write_text('''//! Diagnostic: print every (name, type_name) pair in the
//! Lraspp model's named_descendants_dyn tree. Used to verify the
//! BN-buffer loader can resolve all torchvision state-dict paths.
use ferrotorch_nn::Module;
use ferrotorch_vision::models::segmentation::lraspp_mobilenet_v3_large;

fn main() {
    let model = lraspp_mobilenet_v3_large::<f32>(21).expect("build");
    let model_dyn = &model as &dyn Module<f32>;
    let descendants = model_dyn.named_descendants_dyn();
    for (path, _module) in descendants {
        println!("{path}");
    }
}
''')

# Build & run
subprocess.check_call(
    ['cargo', 'run', '--release', '-p', 'ferrotorch-vision', '--example',
     'dump_lraspp_descendants', '--quiet'],
    stderr=subprocess.DEVNULL,
)
