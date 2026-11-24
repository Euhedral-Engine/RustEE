//! A small JSON reader for `cargo metadata`'s output: objects, arrays, strings, numbers, booleans and null.

use std::collections::BTreeMap;

/// A parsed JSON value. Numbers keep their text; the generator reads none of them as numbers except `cpus`.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Json>),
    Object(BTreeMap<String, Json>),
}

impl Json {
    /// The member `key` of an object, or `None` for a missing member and for any other value.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(map) => map.get(key),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&BTreeMap<String, Json>> {
        match self {
            Json::Object(map) => Some(map),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }
}

/// Parses one JSON document; trailing whitespace is allowed, anything else after the value is an error.
pub fn parse(text: &str) -> Result<Json, String> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        at: 0,
    };
    let value = parser.value()?;
    parser.skip_space();
    if parser.at != parser.bytes.len() {
        return Err(parser.error("trailing characters"));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn error(&self, what: &str) -> String {
        format!("JSON: {what} at byte {}", self.at)
    }

    fn skip_space(&mut self) {
        while let Some(b' ' | b'\n' | b'\r' | b'\t') = self.bytes.get(self.at) {
            self.at += 1;
        }
    }

    fn eat(&mut self, byte: u8) -> Result<(), String> {
        self.skip_space();
        if self.bytes.get(self.at) == Some(&byte) {
            self.at += 1;
            Ok(())
        } else {
            Err(self.error(&format!("expected '{}'", byte as char)))
        }
    }

    fn literal(&mut self, word: &str, value: Json) -> Result<Json, String> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(value)
        } else {
            Err(self.error("unknown literal"))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.skip_space();
        match self.bytes.get(self.at) {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(b'-' | b'0'..=b'9') => Ok(self.number()),
            Some(_) => Err(self.error("unexpected character")),
            None => Err(self.error("unexpected end")),
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.eat(b'{')?;
        let mut map = BTreeMap::new();
        self.skip_space();
        if self.bytes.get(self.at) == Some(&b'}') {
            self.at += 1;
            return Ok(Json::Object(map));
        }
        loop {
            self.skip_space();
            let key = self.string()?;
            self.eat(b':')?;
            let value = self.value()?;
            map.insert(key, value);
            self.skip_space();
            match self.bytes.get(self.at) {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Json::Object(map));
                }
                _ => return Err(self.error("expected ',' or '}'")),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.eat(b'[')?;
        let mut items = Vec::new();
        self.skip_space();
        if self.bytes.get(self.at) == Some(&b']') {
            self.at += 1;
            return Ok(Json::Array(items));
        }
        loop {
            items.push(self.value()?);
            self.skip_space();
            match self.bytes.get(self.at) {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Json::Array(items));
                }
                _ => return Err(self.error("expected ',' or ']'")),
            }
        }
    }

    fn number(&mut self) -> Json {
        let start = self.at;
        while let Some(b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') = self.bytes.get(self.at) {
            self.at += 1;
        }
        Json::Number(String::from_utf8_lossy(&self.bytes[start..self.at]).into_owned())
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let digits = self
            .bytes
            .get(self.at..self.at + 4)
            .and_then(|d| std::str::from_utf8(d).ok())
            .and_then(|d| u32::from_str_radix(d, 16).ok())
            .ok_or_else(|| self.error("bad \\u escape"))?;
        self.at += 4;
        Ok(digits)
    }

    fn string(&mut self) -> Result<String, String> {
        if self.bytes.get(self.at) != Some(&b'"') {
            return Err(self.error("expected a string"));
        }
        self.at += 1;
        let mut out = Vec::new();
        loop {
            match self.bytes.get(self.at) {
                None => return Err(self.error("unterminated string")),
                Some(b'"') => {
                    self.at += 1;
                    return String::from_utf8(out).map_err(|_| self.error("invalid UTF-8"));
                }
                Some(b'\\') => {
                    self.at += 1;
                    let escaped = *self
                        .bytes
                        .get(self.at)
                        .ok_or_else(|| self.error("unterminated escape"))?;
                    self.at += 1;
                    let ch = match escaped {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let high = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&high) {
                                if !self.bytes[self.at..].starts_with(b"\\u") {
                                    return Err(self.error("lone surrogate"));
                                }
                                self.at += 2;
                                let low = self.hex4()?;
                                0x10000
                                    + ((high - 0xD800) << 10)
                                    + (low.wrapping_sub(0xDC00) & 0x3FF)
                            } else {
                                high
                            };
                            char::from_u32(code).ok_or_else(|| self.error("bad code point"))?
                        }
                        _ => return Err(self.error("unknown escape")),
                    };
                    let mut buf = [0; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                Some(&byte) => {
                    out.push(byte);
                    self.at += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_values_and_escapes_parse() {
        let doc = r#" {"a": [1, -2.5e3, true, false, null], "b": {"c": "x\"y\\zé😀\n"}, "d": []} "#;
        let value = parse(doc).unwrap();
        let a = value.get("a").unwrap().as_array().unwrap();
        assert_eq!(a[0], Json::Number("1".into()));
        assert_eq!(a[1], Json::Number("-2.5e3".into()));
        assert_eq!(a[2].as_bool(), Some(true));
        assert!(a[4].is_null());
        let c = value.get("b").and_then(|b| b.get("c")).unwrap();
        assert_eq!(c.as_str(), Some("x\"y\\z\u{e9}\u{1F600}\n"));
        assert_eq!(value.get("d").unwrap().as_array().unwrap().len(), 0);
    }

    #[test]
    fn malformed_documents_are_refused() {
        for doc in ["{", "[1,]", "{\"a\" 1}", "\"open", "1 2", "{\"a\": tru}"] {
            assert!(parse(doc).is_err(), "{doc} parsed");
        }
    }
}
