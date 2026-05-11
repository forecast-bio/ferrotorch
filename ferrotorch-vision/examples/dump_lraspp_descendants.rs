//! Diagnostic (#1146): print every named_descendants_dyn path of the
//! Lraspp model so we can verify that the BN-buffer loader will resolve
//! every torchvision state-dict BN key (e.g. `backbone.features.0.1`).
//! Mirrors the #1142 diagnostic pattern.

use ferrotorch_nn::Module;
use ferrotorch_vision::models::segmentation::lraspp_mobilenet_v3_large;

fn main() {
    let model = lraspp_mobilenet_v3_large::<f32>(21).expect("build");
    let model_dyn = &model as &dyn Module<f32>;
    for (path, _) in model_dyn.named_descendants_dyn() {
        println!("{path}");
    }
}
