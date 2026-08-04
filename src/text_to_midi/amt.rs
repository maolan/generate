//! Anticipatory Music Transformer (AMT) arrival-time tokenization.
//!
//! This is a direct Rust port of the vocabulary layout from the
//! [anticipation](https://github.com/jthickstun/anticipation) Python package,
//! which MIDI-LLM uses to represent MIDI events as token triplets.

/// Time resolution: 100 bins per second (10 ms per bin).
pub const TIME_RESOLUTION: u32 = 100;

/// Maximum sequence length in seconds.
pub const MAX_TIME_IN_SECONDS: u32 = 100;

/// Maximum note duration in seconds.
pub const MAX_DURATION_IN_SECONDS: u32 = 10;

/// Number of MIDI pitch values.
pub const MAX_PITCH: u32 = 128;

/// Number of MIDI instruments (128 melodic + 1 drum kit).
pub const MAX_INSTR: u32 = 129;

/// Maximum onset time in AMT ticks.
pub const MAX_TIME: u32 = TIME_RESOLUTION * MAX_TIME_IN_SECONDS; // 10_000

/// Maximum duration in AMT ticks.
pub const MAX_DUR: u32 = TIME_RESOLUTION * MAX_DURATION_IN_SECONDS; // 1_000

/// Number of distinct instrument-pitch combinations.
pub const MAX_NOTE: u32 = MAX_PITCH * MAX_INSTR; // 16_512

/// Token offsets in the AMT arrival-time vocabulary.
pub const TIME_OFFSET: u32 = 0;
pub const DUR_OFFSET: u32 = TIME_OFFSET + MAX_TIME; // 10_000
pub const NOTE_OFFSET: u32 = DUR_OFFSET + MAX_DUR; // 11_000
pub const REST: u32 = NOTE_OFFSET + MAX_NOTE; // 27_512

/// Anticipated (control) token offsets.
pub const CONTROL_OFFSET: u32 = REST + 1; // 27_513
pub const ATIME_OFFSET: u32 = CONTROL_OFFSET; // 27_513
pub const ADUR_OFFSET: u32 = ATIME_OFFSET + MAX_TIME; // 37_513
pub const ANOTE_OFFSET: u32 = ADUR_OFFSET + MAX_DUR; // 38_513

/// Special tokens.
pub const SPECIAL_OFFSET: u32 = ANOTE_OFFSET + MAX_NOTE; // 55_025
pub const SEPARATOR: u32 = SPECIAL_OFFSET;
pub const AUTOREGRESS: u32 = SPECIAL_OFFSET + 1;
pub const ANTICIPATE: u32 = SPECIAL_OFFSET + 2;

/// Total size of the AMT arrival-time vocabulary.
pub const VOCAB_SIZE: u32 = ANTICIPATE + 1; // 55_028

/// A decoded note event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmtEvent {
    /// Onset time in AMT ticks (10 ms per tick).
    pub time: u32,
    /// Duration in AMT ticks.
    pub duration: u32,
    /// Encoded instrument-pitch token value (raw, without `NOTE_OFFSET`).
    pub note_id: u32,
}

impl AmtEvent {
    /// Instrument index (0..128 melodic, 128 drums).
    pub fn instrument(&self) -> u8 {
        (self.note_id / MAX_PITCH) as u8
    }

    /// MIDI pitch (0..127).
    pub fn pitch(&self) -> u8 {
        (self.note_id % MAX_PITCH) as u8
    }
}

/// Returns true if `token` is a regular (non-control, non-special) time token.
pub const fn is_time_token(token: u32) -> bool {
    token < DUR_OFFSET
}

/// Returns true if `token` is a regular duration token.
pub const fn is_duration_token(token: u32) -> bool {
    token >= DUR_OFFSET && token < NOTE_OFFSET
}

/// Returns true if `token` is a regular instrument-pitch token.
pub const fn is_note_token(token: u32) -> bool {
    token >= NOTE_OFFSET && token < REST
}

/// Returns true if `token` is the rest token.
pub const fn is_rest_token(token: u32) -> bool {
    token == REST
}

/// Returns true if `token` is a control/anticipated token.
pub const fn is_control_token(token: u32) -> bool {
    token >= CONTROL_OFFSET && token < SPECIAL_OFFSET
}

/// Returns true if `token` is one of the special tokens.
pub const fn is_special_token(token: u32) -> bool {
    token >= SPECIAL_OFFSET
}

/// Encode a single event as an AMT token triplet.
pub fn encode_event(event: &AmtEvent) -> [u32; 3] {
    [
        TIME_OFFSET + event.time.min(MAX_TIME - 1),
        DUR_OFFSET + event.duration.min(MAX_DUR - 1),
        NOTE_OFFSET + event.note_id.min(MAX_NOTE - 1),
    ]
}

/// Decode a single AMT token triplet into an event.
///
/// Returns `None` for triplets that do not describe a regular note event
/// (e.g. rests, separators, or control tokens).
pub fn decode_event_triplet(tokens: [u32; 3]) -> Option<AmtEvent> {
    let [time_tok, dur_tok, note_tok] = tokens;
    if !is_time_token(time_tok) || !is_duration_token(dur_tok) || !is_note_token(note_tok) {
        return None;
    }
    Some(AmtEvent {
        time: time_tok - TIME_OFFSET,
        duration: dur_tok - DUR_OFFSET,
        note_id: note_tok - NOTE_OFFSET,
    })
}

/// Convert a flat token sequence into note events, skipping non-event tokens.
pub fn tokens_to_events(tokens: &[u32]) -> Vec<AmtEvent> {
    let mut events = Vec::with_capacity(tokens.len() / 3);
    for i in (0..tokens.len()).step_by(3) {
        if i + 3 > tokens.len() {
            break;
        }
        if let Some(event) = decode_event_triplet([tokens[i], tokens[i + 1], tokens[i + 2]]) {
            events.push(event);
        }
    }
    events
}

/// Convert note events into a flat AMT token sequence.
pub fn events_to_tokens(events: &[AmtEvent]) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(events.len() * 3);
    for event in events {
        tokens.extend_from_slice(&encode_event(event));
    }
    tokens
}

/// Insert separator triplets between logical sequences.
pub fn with_separator(tokens: &mut Vec<u32>) {
    tokens.extend_from_slice(&[SEPARATOR; 3]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_layout_matches_anticipation() {
        assert_eq!(MAX_TIME, 10_000);
        assert_eq!(MAX_DUR, 1_000);
        assert_eq!(MAX_NOTE, 16_512);
        assert_eq!(NOTE_OFFSET, 11_000);
        assert_eq!(REST, 27_512);
        assert_eq!(CONTROL_OFFSET, 27_513);
        assert_eq!(SPECIAL_OFFSET, 55_025);
        assert_eq!(VOCAB_SIZE, 55_028);
    }

    #[test]
    fn round_trip_event() {
        let event = AmtEvent {
            time: 1234,
            duration: 100,
            note_id: 60, // Acoustic Grand Piano, C4
        };
        let triplet = encode_event(&event);
        assert_eq!(triplet[0], TIME_OFFSET + 1234);
        assert_eq!(triplet[1], DUR_OFFSET + 100);
        assert_eq!(triplet[2], NOTE_OFFSET + 60);
        let decoded = decode_event_triplet(triplet).expect("valid event");
        assert_eq!(decoded, event);
    }

    #[test]
    fn decode_ignores_special_tokens() {
        assert!(decode_event_triplet([SEPARATOR, SEPARATOR, SEPARATOR]).is_none());
        assert!(decode_event_triplet([REST, REST, REST]).is_none());
        assert!(
            decode_event_triplet([
                CONTROL_OFFSET,
                CONTROL_OFFSET + MAX_TIME,
                CONTROL_OFFSET + MAX_TIME + MAX_DUR
            ])
            .is_none()
        );
    }

    #[test]
    fn tokens_to_events_filters_non_events() {
        let tokens = vec![
            TIME_OFFSET,
            DUR_OFFSET + 50,
            NOTE_OFFSET + 60,
            SEPARATOR,
            SEPARATOR,
            SEPARATOR,
            TIME_OFFSET + 10,
            DUR_OFFSET + 50,
            NOTE_OFFSET + 64,
        ];
        let events = tokens_to_events(&tokens);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].pitch(), 60);
        assert_eq!(events[1].pitch(), 64);
    }
}
