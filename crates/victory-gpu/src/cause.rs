//! Why a submission sent ahead waited for flights over title memory it
//! reads: which of its steps read what, which flight's declared range was
//! in the way and where that declaration came from. Logged with the
//! frame-rate line, the waits that cost the most first.
//!
//! The frame line already says how often and how long such waits are
//! ("title memory read ... by a fill of upload memory"), and the relay's
//! second stage makes them last only until the flights that declared the
//! memory have landed. Neither tells which declaration it was: a root
//! unordered access view declares all that is left of its buffer, a
//! numbered copy its whole span, so an over-wide declaration and a real
//! shader write look the same there. This keeps, per flight, its declared
//! ranges with their origin (the binding that declared them and the
//! pipeline's shader), and at a wait names the one that held it.
//!
//! It is cheap and on in every build: the declarations are taken from
//! steps the submission walks anyway, and the cause is looked for only
//! when a submission really waits (about once a frame when this was made).
//! It changes nothing a submission does.

use std::sync::{Arc, Mutex};

use crate::pipeline::Shader;

/// What declared a range shaders may write ([`crate::shadow::Shadow::gpu_writes`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Origin {
    /// A root unordered access view: it has no size, so it declares all
    /// that is left of its buffer's copy (`VICTORY_ROOT_UAV_MB` caps it).
    RootUav,
    /// A buffer view a shader can store through, from a descriptor table
    /// or picked from the heap by index: its own range.
    TableUav,
    /// A copy numbered for shaders that reach title memory by address
    /// (`crate::memory`): its whole span.
    Numbered,
}

impl Origin {
    const ALL: [Origin; 3] = [Origin::RootUav, Origin::TableUav, Origin::Numbered];

    fn name(self) -> &'static str {
        match self {
            Origin::RootUav => "root-UAV",
            Origin::TableUav => "table-UAV",
            Origin::Numbered => "numbered copy",
        }
    }

    /// Its bit in [`Cause::origins`].
    fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// Where a declared range came from: the binding and, when the list knew
/// it, the last shader of the pipeline the draw or dispatch used (a
/// container shared by every pipeline with the same bytes, held so the
/// title releasing its pipeline does not matter). It labels the pipeline,
/// not the stage that stores, and pipelines sharing that shader count as
/// one; numbered copies are declared once per list, so theirs is the
/// list's first dispatch that stores by address.
#[derive(Clone)]
pub(crate) struct Declarer {
    pub(crate) origin: Origin,
    pub(crate) by: Option<Arc<Shader>>,
}

impl Declarer {
    fn by_key(&self) -> usize {
        self.by.as_ref().map_or(0, |s| Arc::as_ptr(s) as usize)
    }
}

/// One range a flight's shaders may write, as declared (title addresses,
/// `[from, to)`), and its host copy's number.
#[derive(Clone)]
pub(crate) struct Declaration {
    pub(crate) from: usize,
    pub(crate) to: usize,
    pub(crate) copy: u64,
    pub(crate) declarer: Declarer,
}

/// What kind of step read the title memory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Read {
    /// A title buffer copy (its source or destination).
    Copy,
    /// A fill of the list's upload memory from title memory.
    Fill,
    /// A table's constant buffer view made on a copy in the card's memory
    /// and filled at its submission after all (`Host::place_constants`).
    Constant,
    /// A texture's memory brought into its staging buffer.
    Texture,
}

impl Read {
    fn name(self) -> &'static str {
        match self {
            Read::Copy => "copy",
            Read::Fill => "fill",
            Read::Constant => "constant filled after all",
            Read::Texture => "texture",
        }
    }
}

/// How long a wait lasted, as far as the declarations go.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Until {
    /// Until the named flight had landed (the relay's second stage).
    Declared,
    /// Until the whole line had landed: a step of ours later in the
    /// submission may read or write title memory anywhere, so the wait
    /// lasts for every flight whatever the read touched.
    Run,
    /// Until the whole line had landed: a settle (without the relay's
    /// second stage) waits for every flight.
    Settle,
}

/// Why one wait happened: the step whose read outlasted the others, and
/// the declaration of the last flight in the line that it touched (the
/// flights land in order, so that one is what the wait lasted for, unless
/// [`Cause::until`] says it lasted for the whole line).
#[derive(Clone)]
pub(crate) struct Cause {
    pub(crate) read: Read,
    /// The read range, `[from, to)`.
    pub(crate) at: (usize, usize),
    /// Of that flight's declarations a read touches, the narrowest.
    pub(crate) declared: Declaration,
    /// The origins of all that flight's declarations a read touches
    /// ([`Origin::bit`]): a window both numbered and under a root view's
    /// declaration holds the wait for either, and narrowing one alone
    /// would not end it.
    pub(crate) origins: u8,
    /// How many flights in the line declared something the reads touch.
    pub(crate) flights: u32,
    pub(crate) until: Until,
}

fn overlaps(a: (usize, usize), from: usize, to: usize) -> bool {
    a.0 < to && from < a.1
}

/// The cause of a wait, if any: `reads` are the submission's reading steps
/// (kind, from, to), `flights` the declarations of the flights in the air
/// or being landed, in the line's order.
///
/// The latest flight that touches any read decides how long the wait is;
/// of its declarations that a read touches the narrowest is named (the
/// most specific, whatever order the binding declared them in), with the
/// first read that touches it, and the origins of them all are kept. The
/// reads are joined once so each declaration costs a binary search: this
/// runs on the submitting thread, only when it is about to wait. `until`
/// is how long the wait lasts, which the caller knows.
pub(crate) fn find(reads: &[(Read, usize, usize)], flights: &[&[Declaration]], until: Until) -> Option<Cause> {
    let mut joined: Vec<(usize, usize)> = reads.iter().filter(|r| r.1 < r.2).map(|r| (r.1, r.2)).collect();
    if joined.is_empty() {
        return None;
    }
    joined.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(joined.len());
    for (from, to) in joined {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    let touches = |from: usize, to: usize| {
        let at = out.partition_point(|r| r.1 <= from);
        out.get(at).is_some_and(|r| r.0 < to)
    };
    let mut found: Option<(&Declaration, u8)> = None;
    let mut flights_touching = 0u32;
    for declarations in flights.iter().rev() {
        let mut hits = declarations.iter().filter(|d| d.from < d.to && touches(d.from, d.to));
        let Some(first) = hits.next() else {
            continue;
        };
        flights_touching += 1;
        if found.is_none() {
            found = Some(hits.fold((first, first.declarer.origin.bit()), |(n, o), d| {
                (if d.to - d.from < n.to - n.from { d } else { n }, o | d.declarer.origin.bit())
            }));
        }
    }
    let (declared, origins) = found?;
    let &(read, from, to) = reads.iter().find(|r| overlaps((r.1, r.2), declared.from, declared.to))?;
    Some(Cause {
        read,
        at: (from, to),
        declared: declared.clone(),
        origins,
        flights: flights_touching,
        until,
    })
}

/// Waits of one kind since the last line: same read kind, origin (and
/// the other origins in the way), host copy, shader and how long they
/// lasted. The ranges are the last such wait's (a ring of constants moves
/// every frame; its copy and declarer stay).
struct Entry {
    read: Read,
    origin: Origin,
    origins: u8,
    until: Until,
    copy: u64,
    by: usize,
    label: Option<Arc<Shader>>,
    at: (usize, usize),
    declared: (usize, usize),
    waits: u64,
    us: u64,
    flights: u32,
}

/// Kinds kept per line at most; waits of further kinds count only in the
/// total.
const KINDS: usize = 256;

#[derive(Default)]
struct Causes {
    entries: Vec<Entry>,
    /// Every wait recorded and its microseconds, kept or not.
    waits: u64,
    us: u64,
    /// Of those, the waits no declaration was found for (the line of
    /// flights changed between the look and the wait), and their time.
    unnamed: u64,
    unnamed_us: u64,
}

impl Causes {
    fn add(&mut self, cause: &Cause, us: u64) {
        self.waits += 1;
        self.us += us;
        let d = &cause.declared;
        let by = d.declarer.by_key();
        let kept = self.entries.len();
        let entry = self.entries.iter_mut().find(|e| {
            e.read == cause.read
                && e.origin == d.declarer.origin
                && e.origins == cause.origins
                && e.until == cause.until
                && e.copy == d.copy
                && e.by == by
        });
        match entry {
            Some(e) => {
                e.waits += 1;
                e.us += us;
                e.at = cause.at;
                e.declared = (d.from, d.to);
                e.flights = e.flights.max(cause.flights);
            }
            None if kept < KINDS => self.entries.push(Entry {
                read: cause.read,
                origin: d.declarer.origin,
                origins: cause.origins,
                until: cause.until,
                copy: d.copy,
                by,
                label: d.declarer.by.clone(),
                at: cause.at,
                declared: (d.from, d.to),
                waits: 1,
                us,
                flights: cause.flights,
            }),
            None => {}
        }
    }

    /// A wait of `us` microseconds whose cause was not found.
    fn add_unnamed(&mut self, us: u64) {
        self.waits += 1;
        self.us += us;
        self.unnamed += 1;
        self.unnamed_us += us;
    }

    /// The line for `frames` frames, the eight kinds that waited longest
    /// first; None if nothing waited. Empties the record.
    fn line(&mut self, frames: f64) -> Option<String> {
        let frames = frames.max(1.0);
        let mut entries = std::mem::take(&mut self.entries);
        let (waits, us) = (std::mem::take(&mut self.waits), std::mem::take(&mut self.us));
        let (unnamed, unnamed_us) = (std::mem::take(&mut self.unnamed), std::mem::take(&mut self.unnamed_us));
        if waits == 0 {
            return None;
        }
        entries.sort_by(|a, b| b.us.cmp(&a.us).then(b.waits.cmp(&a.waits)));
        let mut line = format!(
            "settle causes (title memory read behind flights), per frame, the longest waits first: {:.1} ms in {:.1} waits \
             ({:.1} ms in {:.1} not named)",
            us as f64 / 1e3 / frames,
            waits as f64 / frames,
            unnamed_us as f64 / 1e3 / frames,
            unnamed as f64 / frames,
        );
        for e in entries.iter().take(8) {
            let label = match &e.label {
                Some(s) => format!("{} {}", s.stage, s.hash),
                None => "-".to_owned(),
            };
            let also: Vec<&str> = Origin::ALL
                .iter()
                .filter(|o| **o != e.origin && e.origins & o.bit() != 0)
                .map(|o| o.name())
                .collect();
            let also = if also.is_empty() { String::new() } else { format!(" (also {})", also.join(", ")) };
            let until = match e.until {
                Until::Declared => "",
                Until::Run => "; the whole line, for a later step of ours",
                Until::Settle => "; the whole line, settled",
            };
            line.push_str(&format!(
                "; {} {:#x}+{} waits for {} {:#x}+{}{} [copy {}, {}] {:.1} ms/frame ({} waits, up to {} flights{})",
                e.read.name(),
                e.at.0,
                e.at.1 - e.at.0,
                e.origin.name(),
                e.declared.0,
                e.declared.1 - e.declared.0,
                also,
                e.copy,
                label,
                e.us as f64 / 1e3 / frames,
                e.waits,
                e.flights,
                until,
            ));
        }
        Some(line)
    }
}

static CAUSES: Mutex<Option<Causes>> = Mutex::new(None);

/// Counts a wait of `us` microseconds for `cause`, or for none found.
pub(crate) fn record(cause: Option<&Cause>, us: u64) {
    let mut causes = CAUSES.lock().unwrap_or_else(|e| e.into_inner());
    let causes = causes.get_or_insert_with(Causes::default);
    match cause {
        Some(cause) => causes.add(cause, us),
        None => causes.add_unnamed(us),
    }
}

/// For the frame-rate line: the waits since the last one, by cause.
pub(crate) fn line(frames: f64) {
    let line = CAUSES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
        .and_then(|c| c.line(frames));
    if let Some(line) = line {
        tracing::info!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decl(from: usize, to: usize, copy: u64, origin: Origin) -> Declaration {
        Declaration {
            from,
            to,
            copy,
            declarer: Declarer { origin, by: None },
        }
    }

    #[test]
    fn the_latest_flight_touching_a_read_is_named() {
        let first = [decl(0x1000, 0x2000, 1, Origin::TableUav)];
        let second = [decl(0x9000, 0xa000, 2, Origin::TableUav), decl(0x1800, 0x10_0000, 3, Origin::RootUav)];
        let third = [decl(0x20_0000, 0x30_0000, 4, Origin::Numbered)];
        let flights: [&[Declaration]; 3] = [&first, &second, &third];
        let reads = [(Read::Texture, 0x50_0000, 0x50_1000), (Read::Fill, 0x1900, 0x1a00)];
        let cause = find(&reads, &flights, Until::Declared).unwrap();
        assert_eq!(cause.read, Read::Fill);
        assert_eq!(cause.at, (0x1900, 0x1a00));
        assert_eq!(cause.declared.declarer.origin, Origin::RootUav);
        assert_eq!((cause.declared.from, cause.declared.to), (0x1800, 0x10_0000));
        assert_eq!(cause.flights, 2);
        assert_eq!(cause.origins, Origin::RootUav.bit());
    }

    #[test]
    fn every_origin_in_the_way_is_kept_and_the_narrowest_named() {
        // A numbered window declared first, a root view's declaration from
        // inside it to its end second: both hold the wait.
        let flight = [
            decl(0x40_0000, 0x80_0000, 5, Origin::Numbered),
            decl(0x60_0000, 0x80_0000, 5, Origin::RootUav),
            decl(0x90_0000, 0xa0_0000, 6, Origin::TableUav),
        ];
        let flights: [&[Declaration]; 1] = [&flight];
        let cause = find(&[(Read::Fill, 0x70_0000, 0x71_0000)], &flights, Until::Run).unwrap();
        assert_eq!(cause.declared.declarer.origin, Origin::RootUav);
        assert_eq!(cause.origins, Origin::RootUav.bit() | Origin::Numbered.bit());
        assert_eq!(cause.until, Until::Run);
        let mut c = Causes::default();
        c.add(&cause, 10_000);
        let line = c.line(1.0).unwrap();
        let expected = "fill 0x700000+65536 waits for root-UAV 0x600000+2097152 (also numbered copy) [copy 5, -] \
                        10.0 ms/frame (1 waits, up to 1 flights; the whole line, for a later step of ours)";
        assert!(line.contains(expected), "{line}");
    }

    #[test]
    fn no_cause_without_an_overlap() {
        let first = [decl(0x1000, 0x2000, 1, Origin::TableUav)];
        let flights: [&[Declaration]; 1] = [&first];
        // Touching ends do not overlap.
        assert!(find(&[(Read::Copy, 0x2000, 0x3000)], &flights, Until::Declared).is_none());
        assert!(find(&[(Read::Copy, 0x800, 0x1000)], &flights, Until::Declared).is_none());
        assert!(find(&[], &flights, Until::Declared).is_none());
    }

    fn cause(read: Read, at: usize, origin: Origin, copy: u64) -> Cause {
        Cause {
            read,
            at: (at, at + 0x100),
            declared: decl(at, at + 0x1000, copy, origin),
            origins: origin.bit(),
            flights: 1,
            until: Until::Declared,
        }
    }

    #[test]
    fn waits_of_one_kind_are_summed_and_the_longest_lead() {
        let mut c = Causes::default();
        // A moving ring: the same copy and origin at another address each
        // time, one kind with the last ranges.
        c.add(&cause(Read::Fill, 0x1000, Origin::RootUav, 7), 90_000);
        c.add(&cause(Read::Fill, 0x2000, Origin::RootUav, 7), 110_000);
        c.add(&cause(Read::Texture, 0x5000, Origin::Numbered, 9), 5_000);
        c.add(&cause(Read::Fill, 0x5000, Origin::Numbered, 9), 1_000);
        c.add_unnamed(4_000);
        assert_eq!(c.entries.len(), 3);
        let line = c.line(2.0).unwrap();
        assert!(line.contains("105.0 ms in 2.5 waits (2.0 ms in 0.5 not named)"), "{line}");
        let fill = line.find("fill 0x2000+256 waits for root-UAV 0x2000+4096 [copy 7, -] 100.0 ms/frame (2 waits, up to 1 flights)");
        let texture = line.find("texture 0x5000+256 waits for numbered copy");
        assert!(fill.is_some() && texture.is_some(), "{line}");
        assert!(fill < texture, "{line}");
        // The record starts again.
        assert!(c.line(2.0).is_none());
        assert!(c.entries.is_empty());
    }

    #[test]
    fn only_eight_kinds_are_named() {
        let mut c = Causes::default();
        for copy in 0..12u64 {
            c.add(&cause(Read::Fill, 0x1000, Origin::TableUav, copy), (copy + 1) * 1000);
        }
        let line = c.line(1.0).unwrap();
        assert_eq!(line.matches(" waits for ").count(), 8);
        assert!(line.contains("[copy 11, -] 12.0 ms/frame"), "{line}");
        assert!(!line.contains("[copy 3, -]"), "{line}");
    }

    #[test]
    fn kinds_past_the_limit_count_in_the_total_only() {
        let mut c = Causes::default();
        for copy in 0..(KINDS as u64 + 4) {
            c.add(&cause(Read::Copy, 0x1000, Origin::TableUav, copy), 1000);
        }
        assert_eq!(c.entries.len(), KINDS);
        assert_eq!(c.waits, KINDS as u64 + 4);
        let line = c.line(1.0).unwrap();
        assert!(line.contains(&format!("{:.1} ms in {:.1} waits", (KINDS + 4) as f64, (KINDS + 4) as f64)), "{line}");
    }
}
