//! The translator's source fingerprint: the BLAKE3 of every file under
//! `src` (path and content, in path order) and of the features this build
//! has on. `victory-gpu`'s shader cache keys translations by it instead of
//! by the BLAKE3 of the whole executable, so a rebuild that changes no
//! translator source (a runtime change, a new log line) keeps every cached
//! translation, while any change to the translator still makes all of them
//! stale.

use std::path::{Path, PathBuf};

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            files(&path, out);
        } else {
            out.push(path);
        }
    }
}

fn main() {
    let root = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("set by cargo"));
    let src = root.join("src");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=build.rs");
    let mut all = Vec::new();
    files(&src, &mut all);
    let mut named: Vec<(String, PathBuf)> = all
        .into_iter()
        .map(|p| {
            let rel = p
                .strip_prefix(&root)
                .unwrap_or(&p)
                .to_string_lossy()
                .replace('\\', "/");
            (rel, p)
        })
        .collect();
    named.sort();
    let mut h = blake3::Hasher::new();
    for (rel, path) in &named {
        let bytes = std::fs::read(path).expect("a translator source file reads");
        h.update(rel.as_bytes());
        h.update(&[0]);
        h.update(&(bytes.len() as u64).to_le_bytes());
        h.update(&bytes);
    }
    // Features can change what a translation is (`diagnostics` keeps more).
    let mut features: Vec<String> = std::env::vars()
        .filter_map(|(k, _)| k.strip_prefix("CARGO_FEATURE_").map(str::to_owned))
        .collect();
    features.sort();
    for f in &features {
        h.update(b"feature=");
        h.update(f.as_bytes());
        h.update(&[0]);
    }
    let out =
        PathBuf::from(std::env::var_os("OUT_DIR").expect("set by cargo")).join("source_hash.rs");
    let text = format!(
        "/// BLAKE3 of the translator's sources and features (`build.rs`).\npub const SOURCE_HASH: [u8; 32] = {:?};\n",
        h.finalize().as_bytes()
    );
    std::fs::write(out, text).expect("OUT_DIR is writable");
}
