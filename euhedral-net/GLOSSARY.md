# euhedral-net glossary

The implementation-level terms of [`euhedral-net`](README.md). What a user needs to know is in the
[top-level glossary](../GLOSSARY.md). Paths are relative to this crate.

**Head reader** ([`parse`](src/parse.rs)): the part both parsers share. It collects one line at a time, checks the
line against the budget of its section (the start line, or what is left of the header bytes) before it copies the
bytes, and so holds at most a limit's worth. It ends at the empty line.

**Budget**: the bytes a section may still take. The check is on the chunk about to be copied, so the parser fails on
the first byte past a limit and never buffers it.

**Stage**: where a parser is: reading the head, reading a body (with the bytes still pending) or failed. A failed
parser returns its error for every later call and holds nothing.

**Framing**: where a message ends: by `Content-Length` here. Everything strict in the parser is about framing,
because two parsers that frame differently can be made to read one request as two (request smuggling).

**Reserved header**: `Content-Length`, `Transfer-Encoding` and `Connection`, which the writer sets itself; a caller's
line with one of these names is refused ([`write`](src/write.rs)).
