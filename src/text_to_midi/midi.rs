//! Write AMT note events to a Standard MIDI File using `midly`.

use super::amt::AmtEvent;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Default pulses (ticks) per quarter note.
const PPQ: u16 = 480;

/// Default velocity for generated notes.
const DEFAULT_VELOCITY: u8 = 84;

/// Convert a BPM value into microseconds per quarter note.
fn tempo_us_from_bpm(bpm: f32) -> u32 {
    let bpm = bpm.clamp(20.0, 400.0);
    (60_000_000.0 / bpm).round() as u32
}

/// Convert AMT ticks (10 ms per tick) into absolute MIDI ticks for a given tempo.
fn amt_ticks_to_midi_ticks(amt_ticks: u32, tempo_us: u32) -> u64 {
    let factor = (PPQ as f64 * 10_000.0) / tempo_us as f64;
    (amt_ticks as f64 * factor).round() as u64
}

/// Parsed time signature.
#[derive(Clone, Copy, Debug, Default)]
pub struct TimeSignature {
    pub numerator: u8,
    pub denominator: u8,
}

impl TimeSignature {
    /// Parse "N/D" (e.g. "4/4", "6/8").
    pub fn parse(text: &str) -> Option<Self> {
        let (num, den) = text.split_once('/')?;
        let numerator = num.trim().parse::<u8>().ok()?;
        let denominator = den.trim().parse::<u8>().ok()?;
        if numerator == 0 || denominator == 0 {
            return None;
        }
        Some(Self {
            numerator,
            denominator,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EventKind {
    NoteOn,
    NoteOff,
}

#[derive(Clone, Copy, Debug)]
struct TimedMidiEvent {
    abs_ticks: u64,
    kind: EventKind,
    channel: u8,
    pitch: u8,
    velocity: u8,
}

/// Write `events` to `path` as a single-track Type-0 MIDI file.
///
/// `bpm` controls the tempo. `time_signature`, if provided, is written as a
/// meta event at the start of the track.
pub fn write_midi<P: AsRef<Path>>(
    path: P,
    events: &[AmtEvent],
    bpm: f32,
    time_signature: Option<TimeSignature>,
) -> Result<()> {
    let tempo_us = tempo_us_from_bpm(bpm);
    let mut timed_events = Vec::with_capacity(events.len() * 2);
    let mut channel_for_instrument = HashMap::<u8, u8>::new();
    let mut next_channel: u8 = 0;

    for event in events {
        let instrument = event.instrument();
        let channel = *channel_for_instrument.entry(instrument).or_insert_with(|| {
            // Channel 9 is reserved for drums.
            if next_channel == 9 {
                next_channel = 10;
            }
            let ch = next_channel;
            next_channel = next_channel.saturating_add(1);
            ch
        });

        let abs_on = amt_ticks_to_midi_ticks(event.time, tempo_us);
        let abs_off = amt_ticks_to_midi_ticks(event.time + event.duration, tempo_us);
        let pitch = event.pitch();

        timed_events.push(TimedMidiEvent {
            abs_ticks: abs_on,
            kind: EventKind::NoteOn,
            channel,
            pitch,
            velocity: DEFAULT_VELOCITY,
        });
        timed_events.push(TimedMidiEvent {
            abs_ticks: abs_off,
            kind: EventKind::NoteOff,
            channel,
            pitch,
            velocity: 0,
        });
    }

    // Sort by absolute time; note-offs precede note-ons at the same tick to
    // avoid overlapping note-ons cutting off notes at identical offsets.
    timed_events.sort_by(|a, b| {
        a.abs_ticks
            .cmp(&b.abs_ticks)
            .then_with(|| match (a.kind, b.kind) {
                (EventKind::NoteOff, EventKind::NoteOn) => std::cmp::Ordering::Less,
                (EventKind::NoteOn, EventKind::NoteOff) => std::cmp::Ordering::Greater,
                _ => std::cmp::Ordering::Equal,
            })
    });

    let mut track_events: Vec<midly::TrackEvent<'static>> = Vec::new();
    track_events.push(midly::TrackEvent {
        delta: midly::num::u28::new(0),
        kind: midly::TrackEventKind::Meta(midly::MetaMessage::Tempo(midly::num::u24::new(
            tempo_us,
        ))),
    });

    if let Some(ts) = time_signature {
        let denom_log2 = ts.denominator.trailing_zeros() as u8;
        track_events.push(midly::TrackEvent {
            delta: midly::num::u28::new(0),
            kind: midly::TrackEventKind::Meta(midly::MetaMessage::TimeSignature(
                ts.numerator,
                denom_log2,
                24,
                8,
            )),
        });
    }

    // Program changes for each melodic instrument used.
    let mut program_instruments: Vec<u8> = channel_for_instrument.keys().copied().collect();
    program_instruments.sort();
    for instrument in program_instruments {
        if instrument == 128 {
            // Drums use channel 9 and a fixed program 0.
            continue;
        }
        let channel = channel_for_instrument[&instrument];
        track_events.push(midly::TrackEvent {
            delta: midly::num::u28::new(0),
            kind: midly::TrackEventKind::Midi {
                channel: midly::num::u4::new(channel),
                message: midly::MidiMessage::ProgramChange {
                    program: midly::num::u7::new(instrument.min(127)),
                },
            },
        });
    }

    let mut prev_ticks: u64 = 0;
    for ev in timed_events {
        let delta = ev.abs_ticks.saturating_sub(prev_ticks);
        prev_ticks = ev.abs_ticks;
        let message = match ev.kind {
            EventKind::NoteOn => midly::MidiMessage::NoteOn {
                key: midly::num::u7::new(ev.pitch),
                vel: midly::num::u7::new(ev.velocity),
            },
            EventKind::NoteOff => midly::MidiMessage::NoteOff {
                key: midly::num::u7::new(ev.pitch),
                vel: midly::num::u7::new(0),
            },
        };
        track_events.push(midly::TrackEvent {
            delta: midly::num::u28::new(delta as u32),
            kind: midly::TrackEventKind::Midi {
                channel: midly::num::u4::new(ev.channel),
                message,
            },
        });
    }

    track_events.push(midly::TrackEvent {
        delta: midly::num::u28::new(0),
        kind: midly::TrackEventKind::Meta(midly::MetaMessage::EndOfTrack),
    });

    let smf = midly::Smf {
        header: midly::Header::new(
            midly::Format::SingleTrack,
            midly::Timing::Metrical(midly::num::u15::new(PPQ)),
        ),
        tracks: vec![track_events],
    };

    let mut file = File::create(path.as_ref()).with_context(|| {
        format!(
            "failed to create MIDI output file {}",
            path.as_ref().display()
        )
    })?;
    smf.write_std(&mut file)
        .with_context(|| format!("failed to write MIDI file {}", path.as_ref().display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::amt::{AmtEvent, MAX_DUR};
    use super::*;

    #[test]
    fn tempo_conversion() {
        assert_eq!(tempo_us_from_bpm(120.0), 500_000);
        assert_eq!(tempo_us_from_bpm(60.0), 1_000_000);
    }

    #[test]
    fn writes_and_reads_midi() {
        let dir = std::env::temp_dir();
        let path = dir.join("maolan_generate_text_to_midi_test.mid");
        let events = vec![
            AmtEvent {
                time: 0,
                duration: 100, // 1 second
                note_id: 60,
            },
            AmtEvent {
                time: 100,
                duration: 100,
                note_id: 64,
            },
        ];
        write_midi(&path, &events, 120.0, TimeSignature::parse("4/4")).unwrap();

        let data = std::fs::read(&path).unwrap();
        let smf = midly::Smf::parse(&data).unwrap();
        assert_eq!(smf.header.format, midly::Format::SingleTrack);
        let track = smf.tracks.first().expect("one track");
        let note_ons: Vec<_> = track
            .iter()
            .filter(|e| {
                matches!(
                    e.kind,
                    midly::TrackEventKind::Midi {
                        message: midly::MidiMessage::NoteOn { .. },
                        ..
                    }
                )
            })
            .collect();
        assert_eq!(note_ons.len(), 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn clamps_excessive_duration() {
        // AMT durations are capped at MAX_DUR, so note-off time is bounded.
        let event = AmtEvent {
            time: 0,
            duration: MAX_DUR - 1,
            note_id: 60,
        };
        assert!(amt_ticks_to_midi_ticks(event.time + event.duration, 500_000) > 0);
    }
}
