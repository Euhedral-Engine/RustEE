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

**Connection mode** ([`ConnectionMode`](src/message.rs)): what the writer says about the connection: nothing, `keep-alive` or `close`.

**Target form** ([`TargetForm`](src/message.rs)): origin, absolute, authority or asterisk, the four request-target shapes of RFC 9112 section 3.2. The parser checks the form against the method.

**Pending head** ([`PendingHead`](src/parse.rs)): the request head a parser holds while its body is still arriving, readable through `pending_head`.

**Authority** ([`Request::authority`](src/message.rs)): the host and optional port a request is for: the authority of an absolute-form or authority-form target when it has one, else the `Host` value. A server routes on it.

**Interim response**: a 1xx response. The writer never puts a `Connection` line on one, whatever the mode.

**Connection order** ([`connection`](src/connection.rs)): the frames of one connection, all offered with the connection's
id hash, so they run one at a time, in order. The connection's state (parser, buffers, socket, the pending `recv`) is
plain data in a `ClaimCell`; an event that finds it claimed is offered again, never waited for.

**Event**: one thing that happened to a connection (started, bytes or an end arrived, a send ended, a reply was given,
the server stopped), posted as a frame into the connection order. The `Responder` posts the reply the same way.

**Watch `recv`**: the one `recv` kept pending while a request is handled, so that the peer leaving is seen and sets
the request's cancel flag. Bytes it reads are kept for the next request; no new one is issued until the reply is out.

**Linger**: after a final reply that closes the connection, the server shuts its writing side and reads and discards
what the peer sends (up to a budget of the size limits) until the peer closes, a limit passes or the budget is spent.
Closing at once with unread input would reset the connection and could lose the reply.

**Discarded reply**: a reply given for a request whose connection was gone or ended; counted, never an error to the
handler.

**Registry**: the listener state's slab of open connections, used to tell each one of a stop.
