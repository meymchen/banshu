//! A pure, incremental Server-Sent Events decoder.
//!
//! No HTTP dependency: bytes go in via [`SseDecoder::push`] as they arrive off
//! the wire, [`SseEvent`]s come out. [`crate::executor`] is the sole caller,
//! feeding it bytes from a live response body — kept standalone here so it's
//! testable without spinning up a server.

/// One decoded SSE event: the `event:` field (if any) and the `data:` payload,
/// with multiple `data:` lines already joined by `\n`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SseEvent {
    /// The event's `event:` field, if the stream set one.
    pub event: Option<String>,
    /// The joined `data:` payload.
    pub data: String,
}

/// Per-event `data:` accumulation cap. An event that pushes past this is
/// malformed or hostile, not merely large — surfaced as an error rather than
/// silently truncated or buffered without bound.
const MAX_EVENT_DATA_BYTES: usize = 8 * 1024 * 1024;

/// A malformed SSE stream.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum SseError {
    /// A single event's accumulated `data:` exceeded [`MAX_EVENT_DATA_BYTES`].
    #[error("SSE event data exceeded {limit} bytes")]
    EventTooLarge {
        /// The cap that was exceeded.
        limit: usize,
    },
}

/// Incremental SSE decoder.
///
/// Feed raw bytes via [`push`](Self::push) as they arrive; a line or a
/// multi-byte UTF-8 character split across a chunk boundary is buffered until
/// whole (line splitting happens on ASCII `\r` / `\n` bytes, which never occur
/// inside a multi-byte UTF-8 sequence, so decoding a complete line is always
/// safe).
/// Call [`finish`](Self::finish) once at EOF to flush a final event that had
/// no trailing blank line.
#[derive(Debug, Default)]
pub(crate) struct SseDecoder {
    /// Bytes since the last complete line (no line delimiter seen yet).
    pending_line: Vec<u8>,
    /// A chunk ended immediately after a bare `\r` delimiter. If the next
    /// chunk begins with `\n`, it completes that same CRLF delimiter rather
    /// than representing a second blank line.
    discard_leading_lf: bool,
    /// `event:` field for the event currently being assembled.
    event: Option<String>,
    /// `data:` lines for the event currently being assembled, joined on flush.
    data_lines: Vec<String>,
    /// Byte length `data_lines.join("\n")` would have, checked against the cap
    /// as it grows (so it counts the `\n` separators the eventual join adds,
    /// not just the raw line bytes).
    data_len: usize,
    /// Set when an event exceeds the cap, and returned on the *next* call
    /// instead of immediately — so events already decoded earlier in the same
    /// [`push`](Self::push) call are returned via `Ok`, not dropped along with
    /// the error that follows them.
    pending_error: Option<SseError>,
}

impl SseDecoder {
    /// A fresh decoder with no buffered state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a chunk of bytes, returning every event completed by it (usually
    /// zero or one, but a chunk can contain several events back to back).
    ///
    /// If a line in this chunk exceeds the data cap, events already completed
    /// earlier in the same chunk are still returned via `Ok`; the error itself
    /// surfaces on the next call to `push` or `finish`.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, SseError> {
        if let Some(err) = self.pending_error.take() {
            return Err(err);
        }
        let chunk = if self.discard_leading_lf && chunk.first() == Some(&b'\n') {
            self.discard_leading_lf = false;
            &chunk[1..]
        } else {
            if !chunk.is_empty() {
                self.discard_leading_lf = false;
            }
            chunk
        };
        self.pending_line.extend_from_slice(chunk);
        let mut events = Vec::new();
        // Scan forward with a cursor and drain the consumed prefix once at the
        // end, rather than draining per line: draining from the front on every
        // line would shift the whole remaining buffer each time, making a
        // chunk with many lines quadratic instead of linear.
        let mut consumed = 0;
        while let Some(rel_pos) = self.pending_line[consumed..]
            .iter()
            .position(|&byte| matches!(byte, b'\r' | b'\n'))
        {
            let delimiter_pos = consumed + rel_pos;
            let line =
                String::from_utf8_lossy(&self.pending_line[consumed..delimiter_pos]).into_owned();
            let delimiter = self.pending_line[delimiter_pos];
            consumed = delimiter_pos + 1;
            if delimiter == b'\r' {
                if self.pending_line.get(consumed) == Some(&b'\n') {
                    consumed += 1;
                } else if consumed == self.pending_line.len() {
                    self.discard_leading_lf = true;
                }
            }
            match self.process_line(&line) {
                Ok(Some(event)) => events.push(event),
                Ok(None) => {}
                Err(err) => {
                    self.pending_error = Some(err);
                    break;
                }
            }
        }
        self.pending_line.drain(..consumed);
        Ok(events)
    }

    /// Signal EOF: process any unterminated trailing line, then flush a
    /// pending event that never saw its closing blank line.
    pub fn finish(mut self) -> Result<Vec<SseEvent>, SseError> {
        if let Some(err) = self.pending_error.take() {
            return Err(err);
        }
        let mut events = Vec::new();
        if !self.pending_line.is_empty() {
            let mut line = std::mem::take(&mut self.pending_line);
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if let Some(event) = self.process_line(&String::from_utf8_lossy(&line))? {
                events.push(event);
            }
        }
        if let Some(event) = self.flush_event() {
            events.push(event);
        }
        Ok(events)
    }

    /// Apply one already-unterminated SSE line: a blank line dispatches the
    /// pending event, `:`-prefixed lines are comments/keep-alives, and
    /// everything else is a `field:value` (or bare `field`) pair.
    fn process_line(&mut self, line: &str) -> Result<Option<SseEvent>, SseError> {
        if line.is_empty() {
            return Ok(self.flush_event());
        }
        if line.starts_with(':') {
            return Ok(None);
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match field {
            "event" => self.event = Some(value.to_string()),
            "data" => {
                let separator = usize::from(!self.data_lines.is_empty());
                self.data_len += separator + value.len();
                if self.data_len > MAX_EVENT_DATA_BYTES {
                    return Err(SseError::EventTooLarge {
                        limit: MAX_EVENT_DATA_BYTES,
                    });
                }
                self.data_lines.push(value.to_string());
            }
            // `id`, `retry`, and any other field: recognized but not surfaced.
            _ => {}
        }
        Ok(None)
    }

    /// Dispatch the accumulated event, per the SSE spec, only if it collected
    /// at least one `data:` line (a bare `event:` with no data, or a blank
    /// keep-alive line, dispatches nothing).
    fn flush_event(&mut self) -> Option<SseEvent> {
        if self.data_lines.is_empty() {
            self.event = None;
            return None;
        }
        self.data_len = 0;
        Some(SseEvent {
            event: self.event.take(),
            data: std::mem::take(&mut self.data_lines).join("\n"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseResult;

    /// Push a whole payload in one shot and flush EOF, returning all events.
    fn decode_all(input: &[u8]) -> Vec<SseEvent> {
        let mut decoder = SseDecoder::new();
        let mut events = decoder.push(input).expect("push");
        events.extend(decoder.finish().expect("finish"));
        events
    }

    #[derive(Clone, Copy, Debug)]
    enum LineEnding {
        Lf,
        CrLf,
        Cr,
    }

    impl LineEnding {
        fn as_str(self) -> &'static str {
            match self {
                Self::Lf => "\n",
                Self::CrLf => "\r\n",
                Self::Cr => "\r",
            }
        }
    }

    #[derive(Debug)]
    struct EventFixture {
        event: Option<String>,
        data_lines: Vec<String>,
        line_ending: LineEnding,
        include_comment: bool,
    }

    fn event_fixture() -> impl Strategy<Value = EventFixture> {
        (
            prop::option::of("[a-z_]{0,12}"),
            prop::collection::vec("[ -~]{0,24}", 1..5),
            prop_oneof![
                Just(LineEnding::Lf),
                Just(LineEnding::CrLf),
                Just(LineEnding::Cr),
            ],
            any::<bool>(),
        )
            .prop_map(
                |(event, data_lines, line_ending, include_comment)| EventFixture {
                    event,
                    data_lines,
                    line_ending,
                    include_comment,
                },
            )
    }

    fn assert_event_data_is_bounded(events: &[SseEvent]) -> TestCaseResult {
        prop_assert!(
            events
                .iter()
                .all(|event| event.data.len() <= MAX_EVENT_DATA_BYTES)
        );
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn property_decoding_is_independent_of_chunk_boundaries_and_line_endings(
            events in prop::collection::vec(event_fixture(), 0..16),
            chunk_sizes in prop::collection::vec(1_usize..32, 0..64),
        ) {
            let mut wire = String::new();
            let mut expected = Vec::new();
            for fixture in events {
                let ending = fixture.line_ending.as_str();
                if fixture.include_comment {
                    wire.push_str(": provider keep-alive");
                    wire.push_str(ending);
                }
                if let Some(event) = &fixture.event {
                    wire.push_str("event: ");
                    wire.push_str(event);
                    wire.push_str(ending);
                }
                for data in &fixture.data_lines {
                    wire.push_str("data: ");
                    wire.push_str(data);
                    wire.push_str(ending);
                }
                wire.push_str(ending);
                expected.push(SseEvent {
                    event: fixture.event,
                    data: fixture.data_lines.join("\n"),
                });
            }

            let bytes = wire.as_bytes();
            let mut decoder = SseDecoder::new();
            let mut actual = Vec::new();
            let mut offset = 0;
            for size in chunk_sizes {
                if offset == bytes.len() {
                    break;
                }
                let end = (offset + size).min(bytes.len());
                actual.extend(decoder.push(&bytes[offset..end]).expect("bounded valid SSE"));
                offset = end;
            }
            actual.extend(decoder.push(&bytes[offset..]).expect("remaining valid SSE"));
            actual.extend(decoder.finish().expect("finish valid SSE"));

            prop_assert_eq!(actual, expected);
        }

        #[test]
        fn property_bounded_arbitrary_input_never_panics_or_emits_oversized_data(
            input in prop::collection::vec(any::<u8>(), 0..4096),
            chunk_sizes in prop::collection::vec(1_usize..128, 0..64),
        ) {
            let mut decoder = SseDecoder::new();
            let mut offset = 0;
            for size in chunk_sizes {
                if offset == input.len() {
                    break;
                }
                let end = (offset + size).min(input.len());
                if let Ok(events) = decoder.push(&input[offset..end]) {
                    assert_event_data_is_bounded(&events)?;
                }
                offset = end;
            }
            if let Ok(events) = decoder.push(&input[offset..]) {
                assert_event_data_is_bounded(&events)?;
            }
            if let Ok(events) = decoder.finish() {
                assert_event_data_is_bounded(&events)?;
            }
        }
    }

    #[test]
    fn lf_separated_events() {
        let events = decode_all(b"data: hello\n\ndata: world\n\n");
        assert_eq!(
            events,
            vec![
                SseEvent {
                    event: None,
                    data: "hello".to_string()
                },
                SseEvent {
                    event: None,
                    data: "world".to_string()
                },
            ]
        );
    }

    #[test]
    fn crlf_separated_events() {
        let events = decode_all(b"data: hello\r\n\r\ndata: world\r\n\r\n");
        assert_eq!(
            events,
            vec![
                SseEvent {
                    event: None,
                    data: "hello".to_string()
                },
                SseEvent {
                    event: None,
                    data: "world".to_string()
                },
            ]
        );
    }

    #[test]
    fn bare_cr_separated_event_is_decoded_without_leaking_the_delimiter() {
        let events = decode_all(b"data:\r\r");
        assert_eq!(
            events,
            vec![SseEvent {
                event: None,
                data: String::new(),
            }]
        );
    }

    #[test]
    fn multi_line_data_joined_with_lf() {
        let events = decode_all(b"data: line one\ndata: line two\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line one\nline two");
    }

    #[test]
    fn event_field_is_captured() {
        let events = decode_all(b"event: message_start\ndata: {}\n\n");
        assert_eq!(events[0].event.as_deref(), Some("message_start"));
        assert_eq!(events[0].data, "{}");
    }

    #[test]
    fn comment_and_keep_alive_lines_are_ignored() {
        let events = decode_all(b": keep-alive\n\ndata: hello\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn blank_line_with_no_data_dispatches_nothing() {
        let events = decode_all(b"\n\ndata: hello\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn utf8_character_split_across_chunk_boundary() {
        // "你好" splits its first character's 3 UTF-8 bytes across two pushes.
        let payload = "data: 你好\n\n".as_bytes().to_vec();
        let (first, second) = payload.split_at(8);
        let mut decoder = SseDecoder::new();
        let mut events = decoder.push(first).expect("push first");
        events.extend(decoder.push(second).expect("push second"));
        events.extend(decoder.finish().expect("finish"));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "你好");
    }

    #[test]
    fn final_event_without_trailing_blank_line() {
        let events = decode_all(b"data: hello\n\ndata: last");
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].data, "last");
    }

    #[test]
    fn final_event_without_trailing_newline_at_all() {
        let events = decode_all(b"data: last, no newline");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "last, no newline");
    }

    #[test]
    fn events_split_across_pushes() {
        let mut decoder = SseDecoder::new();
        let mut events = decoder.push(b"data: hel").expect("push first");
        assert!(events.is_empty());
        events.extend(decoder.push(b"lo\n\n").expect("push second"));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn crlf_split_across_pushes_is_one_delimiter() {
        let mut decoder = SseDecoder::new();
        assert!(
            decoder
                .push(b"data: hello\r")
                .expect("first push")
                .is_empty()
        );
        let events = decoder.push(b"\n\r\n").expect("second push");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn oversized_event_is_a_protocol_error_not_silence() {
        let mut decoder = SseDecoder::new();
        let big_line = format!("data: {}\n", "a".repeat(MAX_EVENT_DATA_BYTES + 1));
        // The cap trips before any blank line dispatches an event, so this
        // push completes nothing — the error follows on the next call.
        let events = decoder.push(big_line.as_bytes()).expect("push");
        assert!(events.is_empty());
        let err = decoder.finish().unwrap_err();
        assert_eq!(
            err,
            SseError::EventTooLarge {
                limit: MAX_EVENT_DATA_BYTES
            }
        );
    }

    #[test]
    fn events_completed_before_an_oversized_line_are_not_lost() {
        let mut decoder = SseDecoder::new();
        let big = "a".repeat(MAX_EVENT_DATA_BYTES + 1);
        let payload = format!("data: hello\n\ndata: {big}\n");
        let events = decoder.push(payload.as_bytes()).expect("push");
        assert_eq!(
            events,
            vec![SseEvent {
                event: None,
                data: "hello".to_string()
            }]
        );
        let err = decoder.finish().unwrap_err();
        assert_eq!(
            err,
            SseError::EventTooLarge {
                limit: MAX_EVENT_DATA_BYTES
            }
        );
    }

    #[test]
    fn data_under_the_cap_is_accepted() {
        let mut decoder = SseDecoder::new();
        let line = format!("data: {}\n\n", "a".repeat(MAX_EVENT_DATA_BYTES - 1));
        let events = decoder.push(line.as_bytes()).expect("push");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data.len(), MAX_EVENT_DATA_BYTES - 1);
    }

    #[test]
    fn cap_is_cumulative_across_multiple_data_lines() {
        let mut decoder = SseDecoder::new();
        let half = "a".repeat(MAX_EVENT_DATA_BYTES / 2 + 1);
        let payload = format!("data: {half}\ndata: {half}\n");
        let events = decoder.push(payload.as_bytes()).expect("push");
        assert!(events.is_empty());
        let err = decoder.finish().unwrap_err();
        assert_eq!(
            err,
            SseError::EventTooLarge {
                limit: MAX_EVENT_DATA_BYTES
            }
        );
    }

    #[test]
    fn cap_accounts_for_join_separators_between_data_lines() {
        // Many short `data:` lines whose raw bytes sum to well under the cap
        // must still trip it once the `\n`-joined result would exceed it.
        let mut decoder = SseDecoder::new();
        // Each line after the first contributes 1 value byte + 1 separator
        // byte; this many single-byte lines sums under the cap by raw value
        // bytes alone but exceeds it once separators are counted.
        let lines = "data: a\n".repeat(MAX_EVENT_DATA_BYTES / 2 + 2);
        let events = decoder.push(lines.as_bytes()).expect("push");
        assert!(events.is_empty());
        let err = decoder.finish().unwrap_err();
        assert_eq!(
            err,
            SseError::EventTooLarge {
                limit: MAX_EVENT_DATA_BYTES
            }
        );
    }

    #[test]
    fn multiple_events_in_one_chunk() {
        let events = decode_all(b"data: a\n\ndata: b\n\ndata: c\n\n");
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].data, "a");
        assert_eq!(events[1].data, "b");
        assert_eq!(events[2].data, "c");
    }
}
