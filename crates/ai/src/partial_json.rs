//! Best-effort parsing of streamed tool-call arguments.
//!
//! Tool-call arguments arrive as arbitrary fragments of a JSON text. While the
//! call streams, [`parse`] turns the accumulated raw text into a snapshot value
//! after every delta: complete JSON parses exactly (never semantically
//! changed), truncated or mildly malformed JSON is repaired on a best-effort
//! basis (open strings and containers closed, dangling escapes dropped, raw
//! control characters accepted into strings), and anything else is reported as
//! unrepairable so the terminal path can fail loudly instead of fabricating an
//! empty object.

use serde_json::{Map, Value};

/// The outcome of best-effort parsing accumulated tool-call arguments.
pub(crate) enum PartialArguments {
    /// The raw text is complete, valid JSON; the value is the exact parse.
    Complete(Value),
    /// The raw text is truncated or repairable; the value is a best-effort
    /// snapshot (open constructs closed, incomplete trailing fragments
    /// dropped).
    Partial(Value),
    /// The raw text is not JSON and cannot be repaired.
    Invalid,
}

/// Parse accumulated raw arguments into a best-effort snapshot.
///
/// Complete, valid JSON is returned exactly as [`serde_json`] parses it.
/// Anything serde_json rejects is retried with a tolerant parser that only
/// repairs truncation- and escaping-level damage; structural corruption yields
/// [`PartialArguments::Invalid`]. Empty (whitespace-only) input snapshots as an
/// empty object, matching the crate's convention for argument-less calls.
pub(crate) fn parse(raw: &str) -> PartialArguments {
    if let Ok(value) = serde_json::from_str(raw) {
        return PartialArguments::Complete(value);
    }
    if raw.trim().is_empty() {
        return PartialArguments::Partial(Value::Object(Map::new()));
    }
    let mut parser = Parser {
        chars: raw.chars().collect(),
        pos: 0,
    };
    match parser.value() {
        Outcome::Done(value) => {
            parser.skip_ws();
            if parser.at_end() {
                // Only reachable via a repair (an exact parse would have
                // succeeded above), so this is still a best-effort snapshot.
                PartialArguments::Partial(value)
            } else {
                PartialArguments::Invalid
            }
        }
        Outcome::Cut(value) => PartialArguments::Partial(value),
        Outcome::Empty => PartialArguments::Partial(Value::Object(Map::new())),
        Outcome::Bad => PartialArguments::Invalid,
    }
}

/// How a construct ended while parsing.
enum Outcome {
    /// Fully parsed value.
    Done(Value),
    /// The input ended mid-construct; the value is the best-effort close of
    /// everything seen so far.
    Cut(Value),
    /// The input ended before any value started.
    Empty,
    /// Structural corruption that truncation alone cannot explain.
    Bad,
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn at_end(&self) -> bool {
        self.pos >= self.chars.len()
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let ch = self.peek();
        if ch.is_some() {
            self.pos += 1;
        }
        ch
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t' | '\n' | '\r')) {
            self.pos += 1;
        }
    }

    fn value(&mut self) -> Outcome {
        self.skip_ws();
        match self.peek() {
            None => Outcome::Empty,
            Some('{') => {
                self.pos += 1;
                self.object()
            }
            Some('[') => {
                self.pos += 1;
                self.array()
            }
            Some('"') => self.string(),
            Some('t') => self.literal("true", Value::Bool(true)),
            Some('f') => self.literal("false", Value::Bool(false)),
            Some('n') => self.literal("null", Value::Null),
            Some('-') | Some('0'..='9') => self.number(),
            Some(_) => Outcome::Bad,
        }
    }

    /// Parse an object after its `{`. Trailing commas and members cut off at
    /// any point (key, colon, or value) are dropped from the snapshot.
    fn object(&mut self) -> Outcome {
        let mut members = Map::new();
        loop {
            self.skip_ws();
            match self.peek() {
                None => return Outcome::Cut(Value::Object(members)),
                Some('}') => {
                    self.pos += 1;
                    return Outcome::Done(Value::Object(members));
                }
                Some('"') => {}
                Some(_) => return Outcome::Bad,
            }
            let key = match self.string() {
                Outcome::Done(Value::String(key)) => key,
                // A key cut mid-string or followed by anything but `:` is
                // dropped along with any value it might have had.
                Outcome::Cut(_) | Outcome::Empty => return Outcome::Cut(Value::Object(members)),
                Outcome::Bad => return Outcome::Bad,
                Outcome::Done(_) => unreachable!("string() only yields Value::String"),
            };
            self.skip_ws();
            match self.bump() {
                Some(':') => {}
                None => return Outcome::Cut(Value::Object(members)),
                Some(_) => return Outcome::Bad,
            }
            match self.value() {
                Outcome::Done(value) => {
                    members.insert(key, value);
                }
                Outcome::Cut(value) => {
                    members.insert(key, value);
                    return Outcome::Cut(Value::Object(members));
                }
                Outcome::Empty => return Outcome::Cut(Value::Object(members)),
                Outcome::Bad => return Outcome::Bad,
            }
            self.skip_ws();
            match self.bump() {
                Some(',') => continue,
                Some('}') => return Outcome::Done(Value::Object(members)),
                None => return Outcome::Cut(Value::Object(members)),
                Some(_) => return Outcome::Bad,
            }
        }
    }

    /// Parse an array after its `[`. Trailing commas are tolerated; a partial
    /// trailing element is kept in its best-effort form.
    fn array(&mut self) -> Outcome {
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            match self.peek() {
                None => return Outcome::Cut(Value::Array(items)),
                Some(']') => {
                    self.pos += 1;
                    return Outcome::Done(Value::Array(items));
                }
                Some(_) => {}
            }
            match self.value() {
                Outcome::Done(value) => items.push(value),
                Outcome::Cut(value) => {
                    items.push(value);
                    return Outcome::Cut(Value::Array(items));
                }
                Outcome::Empty => return Outcome::Cut(Value::Array(items)),
                Outcome::Bad => return Outcome::Bad,
            }
            self.skip_ws();
            match self.bump() {
                Some(',') => continue,
                Some(']') => return Outcome::Done(Value::Array(items)),
                None => return Outcome::Cut(Value::Array(items)),
                Some(_) => return Outcome::Bad,
            }
        }
    }

    /// Parse a string starting at the opening quote. Unterminated strings
    /// close where the input ends; a dangling escape at the end of input is
    /// dropped; raw control characters and unknown escapes are repaired into
    /// their literal characters.
    fn string(&mut self) -> Outcome {
        debug_assert_eq!(self.peek(), Some('"'));
        self.pos += 1;
        let mut out = String::new();
        loop {
            match self.bump() {
                None => return Outcome::Cut(Value::String(out)),
                Some('"') => return Outcome::Done(Value::String(out)),
                Some('\\') => match self.bump() {
                    None => return Outcome::Cut(Value::String(out)),
                    Some(escape) => match escape {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'b' => out.push('\u{8}'),
                        'f' => out.push('\u{c}'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'u' => match self.unicode_escape() {
                            Some(ch) => out.push(ch),
                            None if self.at_end() => return Outcome::Cut(Value::String(out)),
                            None => return Outcome::Bad,
                        },
                        // Unknown escape: drop the backslash, keep the char.
                        other => out.push(other),
                    },
                },
                Some(ch) => out.push(ch),
            }
        }
    }

    /// Decode a `\uXXXX` sequence (the `\u` is already consumed), combining
    /// surrogate pairs. Returns `None` on invalid hex or truncated input.
    fn unicode_escape(&mut self) -> Option<char> {
        let first = self.hex4()?;
        if (0xD800..0xDC00).contains(&first) {
            // High surrogate: only a following `\uDC00..\uDFFF` completes it.
            let saved = self.pos;
            if self.bump() == Some('\\')
                && self.bump() == Some('u')
                && let Some(second) = self.hex4()
                && (0xDC00..0xE000).contains(&second)
            {
                let combined = 0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00);
                return char::from_u32(combined);
            }
            self.pos = saved;
            Some(char::REPLACEMENT_CHARACTER)
        } else if (0xDC00..0xE000).contains(&first) {
            Some(char::REPLACEMENT_CHARACTER)
        } else {
            char::from_u32(first)
        }
    }

    fn hex4(&mut self) -> Option<u32> {
        let mut value = 0u32;
        for _ in 0..4 {
            let digit = self.bump()?.to_digit(16)?;
            value = value * 16 + digit;
        }
        Some(value)
    }

    /// Parse a number token. A token cut mid-way (`12e`, `1.`, `-`) parses as
    /// far as it validly can; one with nothing valid at all reports [`Outcome::Empty`].
    fn number(&mut self) -> Outcome {
        let start = self.pos;
        while matches!(self.peek(), Some('0'..='9' | '.' | '+' | '-' | 'e' | 'E')) {
            self.pos += 1;
        }
        let mut token: String = self.chars[start..self.pos].iter().collect();
        loop {
            if let Ok(value) = serde_json::from_str::<Value>(&token) {
                return Outcome::Done(value);
            }
            match token.pop() {
                Some('.' | '+' | '-' | 'e' | 'E') => continue,
                Some(_) => return Outcome::Bad,
                None => return Outcome::Empty,
            }
        }
    }

    /// Parse a literal; a prefix cut by the end of input completes to the full
    /// literal (`tru` snapshots as `true`).
    fn literal(&mut self, word: &'static str, value: Value) -> Outcome {
        for expected in word.chars() {
            match self.bump() {
                Some(ch) if ch == expected => {}
                None => return Outcome::Cut(value),
                Some(_) => return Outcome::Bad,
            }
        }
        Outcome::Done(value)
    }
}

#[cfg(test)]
mod tests {
    //! The parser is `pub(crate)`, so its coverage matrix lives inline.
    use super::*;
    use serde_json::json;

    fn complete(raw: &str) -> Value {
        match parse(raw) {
            PartialArguments::Complete(value) => value,
            other => panic!("expected Complete for {raw:?}, got {}", kind(other)),
        }
    }

    fn partial(raw: &str) -> Value {
        match parse(raw) {
            PartialArguments::Partial(value) => value,
            other => panic!("expected Partial for {raw:?}, got {}", kind(other)),
        }
    }

    fn invalid(raw: &str) {
        assert!(
            matches!(parse(raw), PartialArguments::Invalid),
            "expected Invalid for {raw:?}"
        );
    }

    fn kind(outcome: PartialArguments) -> &'static str {
        match outcome {
            PartialArguments::Complete(_) => "Complete",
            PartialArguments::Partial(_) => "Partial",
            PartialArguments::Invalid => "Invalid",
        }
    }

    // -- Legal JSON: exact parse, never semantically changed. --

    #[test]
    fn legal_values_parse_exactly() {
        assert_eq!(
            complete(r#"{"city":"Paris","n":1}"#),
            json!({"city": "Paris", "n": 1})
        );
        assert_eq!(
            complete(r#"[1, "two", null, true]"#),
            json!([1, "two", null, true])
        );
        assert_eq!(complete(r#""just a string""#), json!("just a string"));
        assert_eq!(complete("42"), json!(42));
        assert_eq!(complete("-1.5e3"), json!(-1500.0));
        assert_eq!(
            complete(r#"{"nested":{"a":[{}]},"esc":"\nAé"}"#),
            json!({"nested": {"a": [{}]}, "esc": "\nAé"})
        );
        assert_eq!(complete("{}"), json!({}));
    }

    // -- Unfinished JSON: truncation repaired by closing open constructs. --

    #[test]
    fn unfinished_object_and_array_close() {
        assert_eq!(partial(r#"{"a":1"#), json!({"a": 1}));
        assert_eq!(partial(r#"{"a":1,"b":[2,3"#), json!({"a": 1, "b": [2, 3]}));
        assert_eq!(partial("[1,2"), json!([1, 2]));
        assert_eq!(partial("["), json!([]));
    }

    #[test]
    fn unfinished_members_drop_cleanly() {
        assert_eq!(partial(r#"{"a":"#), json!({}));
        assert_eq!(partial(r#"{"a""#), json!({}));
        assert_eq!(partial(r#"{"a":1,"#), json!({"a": 1}));
        assert_eq!(partial(r#"{"a":1, "b": "#), json!({"a": 1}));
    }

    #[test]
    fn unfinished_string_keeps_its_prefix() {
        assert_eq!(partial(r#"{"city":"Par"#), json!({"city": "Par"}));
        assert_eq!(partial(r#""abc"#), json!("abc"));
    }

    #[test]
    fn unfinished_backslash_constructs_drop() {
        // Dangling escape at end of input.
        assert_eq!(partial(r#"{"a":"x\"#), json!({"a": "x"}));
        // Incomplete \u escape at end of input.
        assert_eq!(partial(r#"{"a":"x\u12"#), json!({"a": "x"}));
    }

    #[test]
    fn unfinished_literals_and_numbers_complete() {
        assert_eq!(partial(r#"{"a":tru"#), json!({"a": true}));
        assert_eq!(partial(r#"{"a":nul"#), json!({"a": null}));
        assert_eq!(partial(r#"{"a":12e"#), json!({"a": 12}));
        assert_eq!(partial(r#"{"a":-"#), json!({}));
    }

    // -- Repairable JSON: valid after escaping-level fixes. --

    #[test]
    fn trailing_commas_are_tolerated() {
        assert_eq!(partial(r#"{"a":1,}"#), json!({"a": 1}));
        assert_eq!(partial("[1,2,]"), json!([1, 2]));
    }

    #[test]
    fn raw_control_characters_in_strings_are_accepted() {
        let raw = "{\"a\":\"line1\nline2\there\"}";
        assert_eq!(partial(raw), json!({"a": "line1\nline2\there"}));
    }

    #[test]
    fn unknown_escapes_keep_the_escaped_character() {
        assert_eq!(partial(r#"{"a":"x\qy"}"#), json!({"a": "xqy"}));
    }

    #[test]
    fn lone_surrogates_become_replacement_characters() {
        assert_eq!(partial(r#"{"a":"\uD800"}"#), json!({"a": "\u{FFFD}"}));
        assert_eq!(complete(r#"{"a":"𝄞"}"#), json!({"a": "𝄞"}));
    }

    // -- Unrepairable JSON: structural corruption. --

    #[test]
    fn unrepairable_inputs_are_invalid() {
        invalid("not json at all");
        invalid("}");
        invalid(r#"{"a":1]"#);
        invalid(r#"{"a":truX}"#);
        invalid(r#"{"a":Paris}"#);
        invalid(r#"{"a":1} trailing"#);
        invalid(r#"{"a" 1}"#);
        invalid("[1,,2]");
        invalid(r#"{"a":"bad\uZZZZ"}"#);
    }

    // -- The streaming progression: snapshots after every delta. --

    #[test]
    fn snapshot_improves_as_deltas_accumulate() {
        let mut raw = String::new();
        for fragment in ["{\"city\":\"", "Par", "is\",\"units\":\"", "metric\"}"] {
            raw.push_str(fragment);
            let snapshot = match parse(&raw) {
                PartialArguments::Complete(value) | PartialArguments::Partial(value) => value,
                PartialArguments::Invalid => panic!("a growing prefix must never be Invalid"),
            };
            // Every snapshot is itself valid, complete JSON.
            assert_eq!(snapshot, complete(&snapshot.to_string()));
        }
        assert_eq!(complete(&raw), json!({"city": "Paris", "units": "metric"}));
    }
}
