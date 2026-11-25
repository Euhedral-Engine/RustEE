//! Writes BUILD files: rule calls with string, list, dict, glob and `select` values, laid out as buildifier does (one
//! attribute per line, lists of two or more entries one per line, trailing commas).

use crate::cfg::Os;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Str(String),
    List(Vec<String>),
    Glob {
        include: Vec<String>,
        exclude: Vec<String>,
        allow_empty: bool,
    },
    Dict(Vec<(String, String)>),
    ListDict(Vec<(String, Vec<String>)>),
    /// A list of labels and another value (`["//:testdata"] + glob([...])`); just the value when there are no labels.
    Plus(Vec<String>, Box<Value>),
    /// A dependency list: entries on every OS, then a `select` over the OSes for the rest (omitted when empty).
    Deps {
        always: Vec<String>,
        by_os: Vec<(Os, Vec<String>)>,
    },
}

pub struct Rule {
    pub kind: &'static str,
    pub attrs: Vec<(&'static str, Value)>,
}

impl Rule {
    pub fn new(kind: &'static str, name: &str) -> Rule {
        Rule {
            kind,
            attrs: vec![("name", Value::Str(name.to_owned()))],
        }
    }

    /// Adds an attribute unless it is empty.
    pub fn attr(&mut self, key: &'static str, value: Value) -> &mut Rule {
        let empty = match &value {
            Value::List(items) => items.is_empty(),
            Value::Dict(items) => items.is_empty(),
            Value::ListDict(items) => items.is_empty(),
            Value::Deps { always, by_os } => {
                always.is_empty() && by_os.iter().all(|(_, d)| d.is_empty())
            }
            Value::Str(_) | Value::Glob { .. } | Value::Plus(..) => false,
        };
        if !empty {
            self.attrs.push((key, value));
        }
        self
    }

    pub fn render(&self) -> String {
        let mut out = format!("{}(\n", self.kind);
        let (name, rest) = self.attrs.split_first().expect("a rule has a name");
        let mut rest: Vec<_> = rest.iter().collect();
        rest.sort_by_key(|(key, _)| *key);
        for (key, value) in std::iter::once(name).chain(rest) {
            out.push_str(&format!("    {key} = {},\n", render(value, 1)));
        }
        out.push_str(")\n");
        out
    }
}

pub fn quote(text: &str) -> String {
    format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""))
}

fn pad(level: usize) -> String {
    "    ".repeat(level)
}

/// A list at nesting `level` (the level of the line it starts on).
pub fn list(items: &[String], level: usize) -> String {
    match items {
        [] => "[]".to_owned(),
        [one] => format!("[{}]", quote(one)),
        _ => {
            let mut out = "[\n".to_owned();
            for item in items {
                out.push_str(&format!("{}{},\n", pad(level + 1), quote(item)));
            }
            out.push_str(&format!("{}]", pad(level)));
            out
        }
    }
}

fn render(value: &Value, level: usize) -> String {
    match value {
        Value::Str(text) => quote(text),
        Value::Plus(labels, rest) if labels.is_empty() => render(rest, level),
        Value::Plus(labels, rest) => format!("{} + {}", list(labels, level), render(rest, level)),
        Value::List(items) => list(items, level),
        Value::Glob {
            include,
            exclude,
            allow_empty,
        } => {
            if exclude.is_empty() && !allow_empty {
                return format!("glob({})", list(include, level));
            }
            let inner = pad(level + 1);
            let mut out = format!("glob(\n{inner}{},\n", list(include, level + 1));
            if !exclude.is_empty() {
                out.push_str(&format!("{inner}exclude = {},\n", list(exclude, level + 1)));
            }
            if *allow_empty {
                out.push_str(&format!("{inner}allow_empty = True,\n"));
            }
            out.push_str(&format!("{})", pad(level)));
            out
        }
        Value::Dict(items) => {
            let mut out = "{\n".to_owned();
            for (key, value) in items {
                out.push_str(&format!(
                    "{}{}: {},\n",
                    pad(level + 1),
                    quote(key),
                    quote(value)
                ));
            }
            out.push_str(&format!("{}}}", pad(level)));
            out
        }
        Value::ListDict(items) => {
            let mut out = "{\n".to_owned();
            for (key, values) in items {
                out.push_str(&format!(
                    "{}{}: {},\n",
                    pad(level + 1),
                    quote(key),
                    list(values, level + 1)
                ));
            }
            out.push_str(&format!("{}}}", pad(level)));
            out
        }
        Value::Deps { always, by_os } => {
            let mut out = String::new();
            if !always.is_empty() || by_os.iter().all(|(_, d)| d.is_empty()) {
                out.push_str(&list(always, level));
            }
            if by_os.iter().any(|(_, d)| !d.is_empty()) {
                if !out.is_empty() {
                    out.push_str(" + ");
                }
                out.push_str("select({\n");
                for (os, deps) in by_os {
                    out.push_str(&format!(
                        "{}{}: {},\n",
                        pad(level + 1),
                        quote(os.constraint()),
                        list(deps, level + 1)
                    ));
                }
                out.push_str(&format!(
                    "{}\"//conditions:default\": [],\n{}}})",
                    pad(level + 1),
                    pad(level)
                ));
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rule_renders_name_first_then_sorted_attributes() {
        let mut rule = Rule::new("rust_test", "x_test");
        rule.attr("tags", Value::List(vec![]))
            .attr(
                "srcs",
                Value::Glob {
                    include: vec!["tests/x.rs".into(), "tests/*/**/*.rs".into()],
                    exclude: vec![],
                    allow_empty: false,
                },
            )
            .attr(
                "deps",
                Value::Deps {
                    always: vec![":a".into()],
                    by_os: vec![
                        (Os::Linux, vec!["@crates//:libc".into()]),
                        (Os::Windows, vec![]),
                    ],
                },
            )
            .attr("edition", Value::Str("2024".into()));
        let want = "rust_test(\n    name = \"x_test\",\n    deps = [\":a\"] + select({\n        \
            \"@platforms//os:linux\": [\"@crates//:libc\"],\n        \"@platforms//os:windows\": [],\n        \
            \"//conditions:default\": [],\n    }),\n    edition = \"2024\",\n    srcs = glob([\n        \
            \"tests/x.rs\",\n        \"tests/*/**/*.rs\",\n    ]),\n)\n";
        assert_eq!(rule.render(), want);
    }
}
