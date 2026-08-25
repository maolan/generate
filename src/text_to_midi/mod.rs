//! Text-to-MIDI generation for `maolan-generate`.
//!
//! This module implements the AMT (Anticipatory Music Transformer) arrival-time
//! tokenization used by MIDI-LLM and provides two generators:
//!
//! - a lightweight, deterministic prompt interpreter (`generator`)
//! - a real MIDI-LLM Llama 3.2 1B model loader and sampler (`midi_llm`)
//!
//! Both generators emit the same AMT token format, so they share the MIDI
//! writer in `midi`.

pub mod amt;
pub mod generator;
pub mod midi;
pub mod midi_llm;
pub mod tiktoken_convert;

pub use amt::{AmtEvent, events_to_tokens, tokens_to_events};
pub use generator::{TextToMidiConfig, generate_events, generate_tokens};
pub use midi::{TimeSignature, write_midi};
pub use midi_llm::{MidiLlmConfig, generate_midi_file as generate_midi_file_with_llm, load_model};

use anyhow::{Context, Result};
use std::path::Path;

/// Generate a MIDI file from `prompt` and write it to `output_path`.
///
/// `bpm`, `key_scale`, and `time_signature` may be extracted from the prompt or
/// supplied explicitly. `length_seconds` caps the generated sequence.
pub fn generate_midi_file<P: AsRef<Path>>(
    output_path: P,
    prompt: &str,
    bpm: Option<f32>,
    key_scale: Option<&str>,
    time_signature: Option<&str>,
    length_seconds: f32,
    seed: u64,
) -> Result<()> {
    let config = TextToMidiConfig {
        prompt,
        bpm,
        key_scale,
        time_signature,
        length_seconds,
        seed,
    };
    let events = generate_events(&config);
    let ts = time_signature.and_then(TimeSignature::parse);
    let effective_bpm = bpm.unwrap_or_else(|| {
        // Re-run lightweight BPM detection when no explicit value was given.
        let words: Vec<String> = prompt
            .to_ascii_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .map(str::to_owned)
            .filter(|w| !w.is_empty())
            .collect();
        generator::detect_bpm(&words, None)
    });
    write_midi(output_path, &events, effective_bpm, ts)
        .with_context(|| "failed to write generated MIDI file")
}
