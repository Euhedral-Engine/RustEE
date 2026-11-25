//! Target `cfg(...)` strings of `[target.'cfg(...)'.dependencies]`, evaluated for the operating systems Bazel builds.

/// The operating systems a dependency can be selected for; `crate_universe` resolves the Linux and Windows triples only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Os {
    Linux = 0,
    Windows = 1,
}

impl Os {
    pub const ALL: [Os; 2] = [Os::Linux, Os::Windows];

    /// The constraint label a `select` names.
    pub fn constraint(self) -> &'static str {
        match self {
            Os::Linux => "@platforms//os:linux",
            Os::Windows => "@platforms//os:windows",
        }
    }

    fn family(self) -> &'static str {
        match self {
            Os::Linux => "unix",
            Os::Windows => "windows",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Os::Linux => "linux",
            Os::Windows => "windows",
        }
    }
}

/// Where a dependency under `target` applies: `None` (no target) is everywhere.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Applies {
    /// On every OS.
    Always,
    /// On these OSes only (possibly none).
    On(Vec<Os>),
}

#[derive(Debug, Clone, PartialEq)]
enum Expr {
    Any(Vec<Expr>),
    All(Vec<Expr>),
    Not(Box<Expr>),
    Name(String),
    KeyValue(String, String),
}

/// Evaluates a dependency's `target` string. A target triple, an unknown name or key, and malformed text are errors: the
/// generator models only what it can map to `@platforms//os`.
pub fn applies(target: Option<&str>) -> Result<Applies, String> {
    let Some(target) = target else {
        return Ok(Applies::Always);
    };
    let inner = target
        .trim()
        .strip_prefix("cfg(")
        .and_then(|rest| rest.strip_suffix(')'))
        .ok_or_else(|| {
            format!("target `{target}` is not a cfg(...): target triples are not modelled")
        })?;
    let mut tokens = tokenize(inner)?;
    tokens.reverse();
    let expr = parse_expr(&mut tokens)?;
    if !tokens.is_empty() {
        return Err(format!("target `{target}`: trailing text"));
    }
    let mut on = Vec::new();
    for os in Os::ALL {
        if eval(&expr, os).map_err(|e| format!("target `{target}`: {e}"))? {
            on.push(os);
        }
    }
    Ok(if on.len() == Os::ALL.len() {
        Applies::Always
    } else {
        Applies::On(on)
    })
}

fn eval(expr: &Expr, os: Os) -> Result<bool, String> {
    Ok(match expr {
        Expr::Any(items) => {
            let mut any = false;
            for item in items {
                any |= eval(item, os)?;
            }
            any
        }
        Expr::All(items) => {
            let mut all = true;
            for item in items {
                all &= eval(item, os)?;
            }
            all
        }
        Expr::Not(item) => !eval(item, os)?,
        Expr::Name(name) => match name.as_str() {
            "unix" | "windows" => name == os.family(),
            // `--cfg loom` is set only by hand for Cargo's model-checking runs: Bazel never builds with it.
            "loom" => false,
            other => return Err(format!("unknown cfg name `{other}`")),
        },
        Expr::KeyValue(key, value) => match key.as_str() {
            "target_os" => value == os.name(),
            "target_family" => value == os.family(),
            other => return Err(format!("unknown cfg key `{other}`")),
        },
    })
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    Str(String),
    Open,
    Close,
    Comma,
    Equals,
}

fn tokenize(text: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' => {
                chars.next();
            }
            '(' => {
                chars.next();
                tokens.push(Token::Open);
            }
            ')' => {
                chars.next();
                tokens.push(Token::Close);
            }
            ',' => {
                chars.next();
                tokens.push(Token::Comma);
            }
            '=' => {
                chars.next();
                tokens.push(Token::Equals);
            }
            '"' => {
                chars.next();
                let mut value = String::new();
                loop {
                    match chars.next() {
                        Some('"') => break,
                        Some(ch) => value.push(ch),
                        None => return Err(format!("cfg `{text}`: unterminated string")),
                    }
                }
                tokens.push(Token::Str(value));
            }
            c if c.is_alphanumeric() || c == '_' => {
                let mut ident = String::new();
                while let Some(&ch) = chars.peek() {
                    if ch.is_alphanumeric() || ch == '_' {
                        ident.push(ch);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Ident(ident));
            }
            other => return Err(format!("cfg `{text}`: unexpected `{other}`")),
        }
    }
    Ok(tokens)
}

/// Parses from a reversed token list (the next token is the last).
fn parse_expr(tokens: &mut Vec<Token>) -> Result<Expr, String> {
    let Some(Token::Ident(ident)) = tokens.pop() else {
        return Err("cfg: expected a name".into());
    };
    match tokens.last() {
        Some(Token::Open) if matches!(ident.as_str(), "any" | "all" | "not") => {
            tokens.pop();
            let mut items = Vec::new();
            if tokens.last() != Some(&Token::Close) {
                loop {
                    items.push(parse_expr(tokens)?);
                    match tokens.pop() {
                        Some(Token::Comma) if tokens.last() == Some(&Token::Close) => {
                            tokens.pop();
                            break;
                        }
                        Some(Token::Comma) => {}
                        Some(Token::Close) => break,
                        _ => return Err("cfg: expected ',' or ')'".into()),
                    }
                }
            } else {
                tokens.pop();
            }
            match ident.as_str() {
                "any" => Ok(Expr::Any(items)),
                "all" => Ok(Expr::All(items)),
                _ => {
                    let [item] = <[Expr; 1]>::try_from(items)
                        .map_err(|_| "cfg: not(...) takes one predicate".to_string())?;
                    Ok(Expr::Not(Box::new(item)))
                }
            }
        }
        Some(Token::Equals) => {
            tokens.pop();
            match tokens.pop() {
                Some(Token::Str(value)) => Ok(Expr::KeyValue(ident, value)),
                _ => Err("cfg: expected a string after '='".into()),
            }
        }
        _ => Ok(Expr::Name(ident)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_workspace_forms_map_to_oses() {
        assert_eq!(applies(None), Ok(Applies::Always));
        assert_eq!(applies(Some("cfg(unix)")), Ok(Applies::On(vec![Os::Linux])));
        assert_eq!(
            applies(Some("cfg(windows)")),
            Ok(Applies::On(vec![Os::Windows]))
        );
        assert_eq!(
            applies(Some(
                r#"cfg(any(target_os = "linux", target_os = "android"))"#
            )),
            Ok(Applies::On(vec![Os::Linux]))
        );
        assert_eq!(
            applies(Some(r#"cfg(not(target_family = "unix"))"#)),
            Ok(Applies::On(vec![Os::Windows]))
        );
        assert_eq!(
            applies(Some(r#"cfg(all(unix, target_os = "macos"))"#)),
            Ok(Applies::On(vec![]))
        );
        // loom is never set under Bazel: a dependency only it needs is nowhere, and one it excludes is everywhere.
        assert_eq!(applies(Some("cfg(loom)")), Ok(Applies::On(vec![])));
        assert_eq!(applies(Some("cfg(not(loom))")), Ok(Applies::Always));
        assert_eq!(
            applies(Some("cfg(any(loom, unix))")),
            Ok(Applies::On(vec![Os::Linux]))
        );
        assert_eq!(
            applies(Some("cfg(any(unix, windows))")),
            Ok(Applies::Always)
        );
    }

    #[test]
    fn unknown_forms_are_refused() {
        for target in [
            "x86_64-pc-windows-msvc",
            r#"cfg(target_arch = "x86_64")"#,
            "cfg(miri)",
            "cfg(not(unix, windows))",
            "cfg(any(unix",
        ] {
            assert!(applies(Some(target)).is_err(), "{target} was accepted");
        }
    }
}
