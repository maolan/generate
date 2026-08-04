//! Prompt-driven deterministic generation of AMT token sequences.
//!
//! This is intentionally model-free: it parses the text prompt for musical
//! hints (tempo, key, instrument, mood) and emits a plausible sequence of AMT
//! events. It is designed to be swapped out later for a neural generator that
//! produces the same AMT token format.

use super::amt::{AmtEvent, MAX_DUR, MAX_TIME, TIME_RESOLUTION};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Generation parameters.
#[derive(Clone, Debug)]
pub struct TextToMidiConfig<'a> {
    pub prompt: &'a str,
    pub bpm: Option<f32>,
    pub key_scale: Option<&'a str>,
    pub time_signature: Option<&'a str>,
    pub length_seconds: f32,
    pub seed: u64,
}

impl<'a> Default for TextToMidiConfig<'a> {
    fn default() -> Self {
        Self {
            prompt: "",
            bpm: None,
            key_scale: None,
            time_signature: None,
            length_seconds: 10.0,
            seed: 0,
        }
    }
}

/// A diatonic scale rooted at a MIDI pitch.
#[derive(Clone, Debug)]
struct Scale {
    root: u8,
    intervals: &'static [i8],
}

impl Scale {
    fn new(root: u8, minor: bool) -> Self {
        const MAJOR: &[i8] = &[0, 2, 4, 5, 7, 9, 11];
        const MINOR: &[i8] = &[0, 2, 3, 5, 7, 8, 10];
        Self {
            root,
            intervals: if minor { MINOR } else { MAJOR },
        }
    }

    fn pitch(&self, degree: usize, octave_offset: i8) -> u8 {
        let degree = degree % self.intervals.len();
        let interval = self.intervals[degree];
        let mut pitch = self.root as i16 + (octave_offset as i16 * 12) + interval as i16;
        pitch = pitch.clamp(0, 127);
        pitch as u8
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum Style {
    #[default]
    Random,
    Arpeggio,
    Chords,
    Bassline,
    Drums,
    Melody,
}

#[derive(Clone, Copy, Debug)]
struct Instrument {
    program: u8,
    base_octave: i8,
    is_drum: bool,
}

fn default_instrument() -> Instrument {
    Instrument {
        program: 0,
        base_octave: 0,
        is_drum: false,
    }
}

fn instrument_from_keyword(word: &str) -> Option<Instrument> {
    match word {
        "piano" | "keyboard" => Some(Instrument {
            program: 0,
            base_octave: 0,
            is_drum: false,
        }),
        "guitar" | "acoustic" => Some(Instrument {
            program: 24,
            base_octave: -1,
            is_drum: false,
        }),
        "electric" | "distortion" => Some(Instrument {
            program: 30,
            base_octave: -1,
            is_drum: false,
        }),
        "bass" | "bassline" => Some(Instrument {
            program: 32,
            base_octave: -2,
            is_drum: false,
        }),
        "drums" | "drum" | "percussion" => Some(Instrument {
            program: 0,
            base_octave: 0,
            is_drum: true,
        }),
        "strings" | "violin" | "cello" => Some(Instrument {
            program: 48,
            base_octave: 0,
            is_drum: false,
        }),
        "synth" | "pad" | "lead" => Some(Instrument {
            program: 80,
            base_octave: 0,
            is_drum: false,
        }),
        "organ" => Some(Instrument {
            program: 16,
            base_octave: 0,
            is_drum: false,
        }),
        "brass" | "trumpet" | "trombone" => Some(Instrument {
            program: 56,
            base_octave: 0,
            is_drum: false,
        }),
        "sax" | "saxophone" => Some(Instrument {
            program: 65,
            base_octave: 0,
            is_drum: false,
        }),
        "flute" | "wind" => Some(Instrument {
            program: 73,
            base_octave: 0,
            is_drum: false,
        }),
        _ => None,
    }
}

fn style_from_keyword(word: &str) -> Option<Style> {
    match word {
        "arpeggio" | "arp" => Some(Style::Arpeggio),
        "chords" | "chord" | "harmony" => Some(Style::Chords),
        "bassline" | "bass" => Some(Style::Bassline),
        "drums" | "drum" | "percussion" => Some(Style::Drums),
        "melody" | "lead" | "solo" => Some(Style::Melody),
        _ => None,
    }
}

fn parse_key_scale(text: &str) -> Option<Scale> {
    let mut parts = text.split_whitespace();
    let root_str = parts.next()?;
    let mode = parts.next()?.to_ascii_lowercase();
    let minor = mode == "minor" || mode == "min";
    if mode != "major" && mode != "maj" && !minor {
        return None;
    }
    let root = parse_root(root_str)?;
    Some(Scale::new(root, minor))
}

fn parse_root(text: &str) -> Option<u8> {
    let mut chars = text.chars();
    let letter = chars.next()?.to_ascii_uppercase();
    let accidental: String = chars.collect();
    let semitone = match letter {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => return None,
    };
    let offset = match accidental.as_str() {
        "" | "natural" => 0,
        "#" | "sharp" => 1,
        "b" | "flat" | "♭" => -1,
        _ => return None,
    };
    let mut pitch = semitone as i16 + offset as i16;
    pitch = pitch.rem_euclid(12);
    Some((pitch + 60) as u8) // root around C4
}

fn normalize_prompt(text: &str) -> Vec<String> {
    text.to_ascii_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .map(str::to_owned)
        .filter(|w| !w.is_empty())
        .collect()
}

pub(crate) fn detect_bpm(words: &[String], override_bpm: Option<f32>) -> f32 {
    if let Some(bpm) = override_bpm {
        return bpm.clamp(20.0, 400.0);
    }
    for (i, word) in words.iter().enumerate() {
        if word == "bpm"
            && i > 0
            && let Ok(bpm) = words[i - 1].parse::<f32>()
        {
            return bpm.clamp(20.0, 400.0);
        }
    }
    for word in words {
        match word.as_str() {
            "fast" | "upbeat" | "energetic" | "driving" => return 140.0,
            "slow" | "calm" | "relaxing" | "ambient" => return 70.0,
            "medium" | "moderate" => return 110.0,
            _ => {}
        }
    }
    120.0
}

fn detect_instrument(words: &[String], style: Style) -> Instrument {
    if style == Style::Drums {
        return Instrument {
            program: 0,
            base_octave: 0,
            is_drum: true,
        };
    }
    for word in words {
        if let Some(inst) = instrument_from_keyword(word) {
            return inst;
        }
    }
    default_instrument()
}

fn detect_style(words: &[String], instrument: Instrument) -> Style {
    if instrument.is_drum {
        return Style::Drums;
    }
    for word in words {
        if let Some(style) = style_from_keyword(word) {
            return style;
        }
    }
    Style::Random
}

fn parse_time_signature(text: Option<&str>) -> (u8, u8) {
    let Some(text) = text else {
        return (4, 4);
    };
    let Some((num, den)) = text.split_once('/') else {
        return (4, 4);
    };
    let Ok(num) = num.trim().parse::<u8>() else {
        return (4, 4);
    };
    let Ok(den) = den.trim().parse::<u8>() else {
        return (4, 4);
    };
    if num == 0 || den == 0 {
        return (4, 4);
    }
    (num, den)
}

fn drum_pitch(drum_step: usize) -> u8 {
    match drum_step % 4 {
        0 => 36, // kick
        2 => 38, // snare
        _ => 42, // closed hi-hat
    }
}

fn beat_step_ticks(bpm: f32, denominator: u8) -> f64 {
    let quarter_seconds = 60.0 / bpm as f64;
    let beat_value = match denominator {
        8 => 0.5, // eighth note grid
        2 => 2.0, // half note grid
        _ => 1.0, // quarter note grid
    };
    quarter_seconds * beat_value * TIME_RESOLUTION as f64
}

/// Generate a deterministic AMT event sequence from the prompt.
pub fn generate_events(config: &TextToMidiConfig<'_>) -> Vec<AmtEvent> {
    let words = normalize_prompt(config.prompt);
    let bpm = detect_bpm(&words, config.bpm);
    let scale = config
        .key_scale
        .and_then(parse_key_scale)
        .unwrap_or_else(|| Scale::new(60, false)); // C major
    let (numerator, denominator) = parse_time_signature(config.time_signature);
    let step_ticks = beat_step_ticks(bpm, denominator);
    let length_ticks =
        ((config.length_seconds * TIME_RESOLUTION as f32).round() as u32).min(MAX_TIME - 1);
    let total_steps = ((length_ticks as f64) / step_ticks.max(1.0)).floor() as usize;

    let mut rng = StdRng::seed_from_u64(config.seed);
    let instrument = detect_instrument(&words, Style::default());
    let style = detect_style(&words, instrument);

    let mut events = Vec::new();

    match style {
        Style::Drums => {
            let step_ticks_int = step_ticks.round().max(1.0) as u32;
            for step in 0..total_steps {
                let time = (step as u32)
                    .saturating_mul(step_ticks_int)
                    .min(length_ticks);
                if time >= length_ticks {
                    break;
                }
                let pitch = drum_pitch(step);
                events.push(AmtEvent {
                    time,
                    duration: (step_ticks_int / 4).max(1),
                    note_id: 128_u32 * 128 + u32::from(pitch),
                });
            }
        }
        Style::Bassline => {
            let step_ticks_int = step_ticks.round().max(1.0) as u32;
            for step in 0..total_steps {
                let time = (step as u32)
                    .saturating_mul(step_ticks_int)
                    .min(length_ticks);
                if time >= length_ticks {
                    break;
                }
                let is_downbeat = step % numerator as usize == 0;
                let play = is_downbeat || rng.random::<f32>() < 0.25;
                if !play {
                    continue;
                }
                let degree = if is_downbeat {
                    0
                } else {
                    rng.random_range(0..7)
                };
                let pitch = scale.pitch(degree, instrument.base_octave);
                let duration = if is_downbeat {
                    step_ticks_int * 2
                } else {
                    step_ticks_int
                };
                events.push(AmtEvent {
                    time,
                    duration: duration.min(MAX_DUR - 1),
                    note_id: u32::from(instrument.program) * 128 + u32::from(pitch),
                });
            }
        }
        Style::Chords => {
            let step_ticks_int = step_ticks.round().max(1.0) as u32;
            let bar_ticks = step_ticks_int.saturating_mul(u32::from(numerator));
            for bar in 0..total_steps.saturating_div(numerator as usize).max(1) {
                let time = (bar as u32).saturating_mul(bar_ticks).min(length_ticks);
                if time >= length_ticks {
                    break;
                }
                let duration = bar_ticks.min(MAX_DUR - 1);
                for offset in [0, 2, 4] {
                    let pitch = scale.pitch(offset, instrument.base_octave);
                    events.push(AmtEvent {
                        time,
                        duration,
                        note_id: u32::from(instrument.program) * 128 + u32::from(pitch),
                    });
                }
            }
        }
        Style::Arpeggio => {
            let step_ticks_int = step_ticks.round().max(1.0) as u32;
            for step in 0..total_steps {
                let time = (step as u32)
                    .saturating_mul(step_ticks_int)
                    .min(length_ticks);
                if time >= length_ticks {
                    break;
                }
                let degree = [0, 2, 4, 7][step % 4];
                let pitch = scale.pitch(degree, instrument.base_octave);
                events.push(AmtEvent {
                    time,
                    duration: step_ticks_int * 2,
                    note_id: u32::from(instrument.program) * 128 + u32::from(pitch),
                });
            }
        }
        Style::Melody | Style::Random => {
            let step_ticks_int = step_ticks.round().max(1.0) as u32;
            let density = if words
                .iter()
                .any(|w| matches!(w.as_str(), "dense" | "busy" | "fast"))
            {
                0.7
            } else if words
                .iter()
                .any(|w| matches!(w.as_str(), "sparse" | "minimal"))
            {
                0.25
            } else {
                0.45
            };
            for step in 0..total_steps {
                let time = (step as u32)
                    .saturating_mul(step_ticks_int)
                    .min(length_ticks);
                if time >= length_ticks {
                    break;
                }
                if rng.random::<f32>() > density {
                    continue;
                }
                let degree = rng.random_range(0..7);
                let octave = instrument.base_octave + rng.random_range(0..2);
                let pitch = scale.pitch(degree, octave);
                let duration = step_ticks_int * rng.random_range(1..=3);
                events.push(AmtEvent {
                    time,
                    duration: duration.min(MAX_DUR - 1),
                    note_id: u32::from(instrument.program) * 128 + u32::from(pitch),
                });
            }
        }
    }

    events.sort_by_key(|e| e.time);
    events
}

/// Generate a flat AMT token sequence from the prompt.
pub fn generate_tokens(config: &TextToMidiConfig<'_>) -> Vec<u32> {
    super::amt::events_to_tokens(&generate_events(config))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_generation_produces_events() {
        let config = TextToMidiConfig {
            prompt: "happy piano melody",
            ..Default::default()
        };
        let events = generate_events(&config);
        assert!(!events.is_empty(), "should generate at least one note");
        for event in &events {
            assert!(event.pitch() <= 127);
            assert!(event.duration > 0);
        }
    }

    #[test]
    fn drum_style_generates_channel_9_events() {
        let config = TextToMidiConfig {
            prompt: "fast drum beat",
            length_seconds: 2.0,
            ..Default::default()
        };
        let events = generate_events(&config);
        assert!(events.iter().all(|e| e.instrument() == 128));
    }

    #[test]
    fn key_scale_override_used() {
        let config = TextToMidiConfig {
            prompt: "bassline",
            key_scale: Some("A minor"),
            length_seconds: 1.0,
            ..Default::default()
        };
        let events = generate_events(&config);
        assert!(!events.is_empty());
    }

    #[test]
    fn seed_is_deterministic() {
        let config = TextToMidiConfig {
            prompt: "melody",
            seed: 42,
            ..Default::default()
        };
        let first = generate_events(&config);
        let second = generate_events(&config);
        assert_eq!(first, second);
    }
}
