//! The account: where the CPU's work on title memory and the process's
//! memory go, logged with the frame-rate line every five seconds in every
//! build, so that a run tells without guessing which memory is copied up,
//! compared, merged back or fetched, of what kind, and how much of it the
//! title's CPU or the GPU actually writes.
//!
//! The frame-rate line's own buckets are taken every second and logged
//! every five, so what it says of bytes is the last second's. The account
//! keeps its own counters and sums the whole five seconds: bytes by what
//! was done ([`Op`]) and the kind of memory it touched ([`Mem`], [`Side`]);
//! per host copy, its bytes and the frames the CPU or the GPU wrote it
//! ([`Tally`]: the ten copies with the most CPU bytes are named); what our
//! layer holds right now; and the process's memory and page faults.
//!
//! `VICTORY_ACCOUNT_CSV=<file>` also appends one row per period to that
//! file (a header first if the file is empty): every number of the lines,
//! for a run's whole history in a spreadsheet.
//!
//! Cost: a relaxed atomic add where the bytes are counted already (per
//! copy, per compared range, per merged piece; never per byte), two clock
//! reads around each refresh copy, and once per five seconds a walk of the
//! host copies' tables and two process queries.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering::Relaxed};

/// What was done with bytes of title memory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Op {
    /// A refresh was asked to bring these bytes of a host copy up to date
    /// (what the write watch and the guards then cut down to `Copied`).
    Asked,
    /// Refresh: pages the title wrote since the copy last took them (or
    /// never taken), copied into the copy.
    Copied,
    /// Refresh of memory neither write-watched nor a view: copied whole,
    /// every time.
    Whole,
    /// Title memory read and compared, page by page, under ranges shaders
    /// of a flight in the air may be writing, so that the title's changes
    /// go into the copy without settling the flight
    /// (`Shadow::merge_under_writes`).
    Under,
    /// Copied into a snapshot before shaders that may write them ran.
    Snapshot,
    /// Compared after the card ran: with the snapshot on the CPU, or by our
    /// compare pass on the card for copies in the card's memory.
    Compared,
    /// In 64 KiB pieces found differing by that compare: for copies in the
    /// card's memory, the pieces it flagged (cut to the declared span),
    /// fetched; pieces of ranges it did not compare are fetched but not
    /// counted here.
    Differing,
    /// Written into title memory by the merge: from the first to the last
    /// byte of each piece where the GPU's bytes went in.
    Merged,
    /// Sent from a card copy's CPU side up to its card side.
    SentUp,
    /// Fetched down from card copies (the pieces their compare flagged).
    Fetched,
    /// Texture refresh: title bytes read (through the kernel or in place).
    TexRead,
    /// Texture refresh: title bytes hashed.
    TexHashed,
    /// Texture refresh: title bytes detiled into staging.
    TexDetiled,
    /// Our own title buffer copies (`Step::Copy`): title bytes read and
    /// written into title memory.
    StepCopy,
    /// Our own fills of upload memory from title memory (`Step::Fill`).
    StepFill,
}

pub(crate) const OPS: usize = 15;
const OP_ALL: [Op; OPS] = [
    Op::Asked,
    Op::Copied,
    Op::Whole,
    Op::Under,
    Op::Snapshot,
    Op::Compared,
    Op::Differing,
    Op::Merged,
    Op::SentUp,
    Op::Fetched,
    Op::TexRead,
    Op::TexHashed,
    Op::TexDetiled,
    Op::StepCopy,
    Op::StepFill,
];

impl Op {
    fn name(self) -> &'static str {
        match self {
            Op::Asked => "asked",
            Op::Copied => "copied",
            Op::Whole => "whole",
            Op::Under => "under_flights",
            Op::Snapshot => "snapshotted",
            Op::Compared => "compared",
            Op::Differing => "differing",
            Op::Merged => "merged",
            Op::SentUp => "sent_up",
            Op::Fetched => "fetched",
            Op::TexRead => "tex_read",
            Op::TexHashed => "tex_hashed",
            Op::TexDetiled => "tex_detiled",
            Op::StepCopy => "step_copy",
            Op::StepFill => "step_fill",
        }
    }
}

/// The kind of title memory: as the write watch and the guards found it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mem {
    /// Not asked yet (or not asked at all: our own copies and fills).
    Unknown = 0,
    /// Write-watched memory (the kernel's dirty bits).
    Watched = 1,
    /// View memory (physical-page views and their placeholders): guarded.
    View = 2,
    /// Neither: copied whole at every refresh, or hashed.
    Neither = 3,
}

pub(crate) const MEMS: usize = 4;
const MEM_NAMES: [&str; MEMS] = ["unknown", "watched", "view", "neither"];

impl Mem {
    fn from_u8(v: u8) -> Mem {
        match v {
            1 => Mem::Watched,
            2 => Mem::View,
            3 => Mem::Neither,
            _ => Mem::Unknown,
        }
    }
}

/// Where the host's side of the bytes is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Side {
    /// A title buffer's copy in the PC's memory.
    PcBuffer = 0,
    /// A window of loose title memory, copied in the PC's memory.
    PcWindow = 1,
    /// A title buffer's copy in the card's memory (CPU side in the PC's).
    CardBuffer = 2,
    /// A window in the card's memory (`VICTORY_CARD_WINDOWS_MB`).
    CardWindow = 3,
    /// Title memory used in place (`VICTORY_INPLACE`): nothing copied.
    InPlace = 4,
    /// A texture's staging buffer.
    Texture = 5,
    /// No host copy: title memory to title memory or to upload memory.
    Direct = 6,
}

pub(crate) const SIDES: usize = 7;
const SIDE_NAMES: [&str; SIDES] = ["pc_buffer", "pc_window", "card_buffer", "card_window", "in_place", "texture", "direct"];

impl Side {
    #[cfg(test)]
    fn from_index(i: usize) -> Side {
        match i {
            0 => Side::PcBuffer,
            1 => Side::PcWindow,
            2 => Side::CardBuffer,
            3 => Side::CardWindow,
            4 => Side::InPlace,
            5 => Side::Texture,
            _ => Side::Direct,
        }
    }

    /// Whether the compares of such a copy run on the CPU.
    fn compares_on_cpu(self) -> bool {
        !matches!(self, Side::CardBuffer | Side::CardWindow)
    }
}

const CELLS: usize = OPS * MEMS * SIDES;

fn cell(op: Op, mem: Mem, side: Side) -> usize {
    (op as usize * MEMS + mem as usize) * SIDES + side as usize
}

/// Bytes since the last account, by operation, kind of memory and side.
static BYTES: [AtomicU64; CELLS] = [const { AtomicU64::new(0) }; CELLS];

/// Microseconds the refresh copies themselves took (the reads of title
/// memory into copies, not the write-watch askings around them), since the
/// last account: what [`Op::Copied`] and [`Op::Whole`] took.
static COPY_US: AtomicU64 = AtomicU64::new(0);

/// Counts `bytes` of `op` on memory of kinds `mem` and `side`.
pub(crate) fn add(op: Op, mem: Mem, side: Side, bytes: u64) {
    if bytes != 0 {
        BYTES[cell(op, mem, side)].fetch_add(bytes, Relaxed);
    }
}

/// A refresh copy took `us` microseconds.
pub(crate) fn copy_took(us: u64) {
    COPY_US.fetch_add(us, Relaxed);
}

/// The refresh copy time counted so far in this period.
pub(crate) fn copy_us() -> u64 {
    COPY_US.load(Relaxed)
}

/// Refreshes ran on several threads at once: `copying` microseconds of
/// copies were counted within `summed` microseconds of the threads'
/// refreshes, which took `elapsed` of the frame. The copies keep their
/// share of that time and give back the rest.
pub(crate) fn copy_wall(copying: u64, summed: u64, elapsed: u64) {
    let extra = copying - wall_share(copying, summed, elapsed);
    if extra > 0 {
        COPY_US.fetch_sub(extra.min(COPY_US.load(Relaxed)), Relaxed);
    }
}

/// `copying` of `summed` threads' microseconds, as a share of `elapsed`
/// (unchanged when the threads did not overlap).
fn wall_share(copying: u64, summed: u64, elapsed: u64) -> u64 {
    if summed <= elapsed {
        return copying;
    }
    (u128::from(copying) * u128::from(elapsed) / u128::from(summed)) as u64
}

/// The account starts here (the first present): what was counted before,
/// while the title loaded, is dropped, so that the first period holds only
/// its own frames' work. Copies' tallies are emptied with one walk of the
/// tables, and the page fault count is taken as the starting point.
pub(crate) fn start() {
    for b in &BYTES {
        b.store(0, Relaxed);
    }
    COPY_US.store(0, Relaxed);
    drop(crate::shadow::account_walk());
    *LAST_FAULTS.lock().unwrap_or_else(|e| e.into_inner()) = Some(process_memory().2);
}

/// Frames presented so far, from 1: what [`Tally`] stamps frames with.
static FRAME: AtomicU32 = AtomicU32::new(1);

/// A frame was presented.
pub(crate) fn presented() {
    FRAME.fetch_add(1, Relaxed);
}

/// Bytes held now in snapshots of copies (`Shadow::before`), and in our
/// threads' texture read buffers (each grows to the biggest texture its
/// thread read, and stays).
static SNAPSHOT_HELD: AtomicU64 = AtomicU64::new(0);
static TEXTURE_READ_HELD: AtomicU64 = AtomicU64::new(0);

/// A snapshot buffer of `bytes` is held (or let go, `held` false).
pub(crate) fn snapshot_held(bytes: usize, held: bool) {
    if held {
        SNAPSHOT_HELD.fetch_add(bytes as u64, Relaxed);
    } else {
        SNAPSHOT_HELD.fetch_sub(bytes as u64, Relaxed);
    }
}

/// A thread's texture read buffer grew by `bytes`.
pub(crate) fn texture_read_grew(bytes: usize) {
    TEXTURE_READ_HELD.fetch_add(bytes as u64, Relaxed);
}

/// Per host copy, since the last account: its CPU-side bytes and the
/// frames the CPU and the GPU wrote it. Kept in the copy, taken by the
/// walk over the copies' tables at each account.
#[derive(Default)]
pub(crate) struct Tally {
    /// Copied up (refreshes of all kinds), compared, found differing,
    /// merged back, sent up to the card side.
    bytes: [AtomicU64; 5],
    /// Frames in which the write watch or the guards found the title's
    /// writes in it (beside the first copy), in which shaders were declared
    /// to write it, and in which our merge wrote title memory under it;
    /// with the frame each was last counted in.
    frames: [AtomicU32; 3],
    seen: [AtomicU32; 3],
    /// The kind of its memory, as the last refresh found it ([`Mem`]).
    mem: AtomicU8,
}

/// The frames [`Tally::mark`] counts.
#[derive(Clone, Copy)]
pub(crate) enum Wrote {
    Cpu = 0,
    Gpu = 1,
    Merge = 2,
}

impl Tally {
    pub(crate) fn add(&self, op: Op, bytes: u64) {
        let i = match op {
            Op::Copied | Op::Whole | Op::Under => 0,
            Op::Compared => 1,
            Op::Differing => 2,
            Op::Merged => 3,
            Op::SentUp => 4,
            _ => return,
        };
        if bytes != 0 {
            self.bytes[i].fetch_add(bytes, Relaxed);
        }
    }

    /// The copy was written this frame, by `who` (counted once a frame).
    pub(crate) fn mark(&self, who: Wrote) {
        let (i, frame) = (who as usize, FRAME.load(Relaxed));
        if self.seen[i].load(Relaxed) != frame && self.seen[i].swap(frame, Relaxed) != frame {
            self.frames[i].fetch_add(1, Relaxed);
        }
    }

    pub(crate) fn mem(&self) -> Mem {
        Mem::from_u8(self.mem.load(Relaxed))
    }

    pub(crate) fn set_mem(&self, mem: Mem) {
        self.mem.store(mem as u8, Relaxed);
    }

    /// The counts since the last take, as a row for the range
    /// `[start, start + len)` on `side`; `None` if it did nothing.
    pub(crate) fn take(&self, start: u64, len: u64, side: Side) -> Option<Row> {
        let b = [0, 1, 2, 3, 4].map(|i| self.bytes[i].swap(0, Relaxed));
        let f = [0, 1, 2].map(|i| self.frames[i].swap(0, Relaxed));
        if b.iter().all(|&v| v == 0) && f.iter().all(|&v| v == 0) {
            return None;
        }
        Some(Row {
            start,
            len,
            side,
            mem: self.mem(),
            copied: b[0],
            compared: b[1],
            differing: b[2],
            merged: b[3],
            sent: b[4],
            cpu_frames: f[0],
            gpu_frames: f[1],
            merge_frames: f[2],
        })
    }
}

/// One host copy's period, for the top ranges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Row {
    pub(crate) start: u64,
    pub(crate) len: u64,
    pub(crate) side: Side,
    pub(crate) mem: Mem,
    pub(crate) copied: u64,
    pub(crate) compared: u64,
    pub(crate) differing: u64,
    pub(crate) merged: u64,
    pub(crate) sent: u64,
    pub(crate) cpu_frames: u32,
    pub(crate) gpu_frames: u32,
    pub(crate) merge_frames: u32,
}

impl Row {
    /// The CPU's bytes on it: copied up, merged back, and compared where
    /// the CPU compares (the card compares its own copies).
    fn cpu_bytes(&self) -> u64 {
        self.copied + self.merged + if self.side.compares_on_cpu() { self.compared } else { 0 }
    }
}

/// How many ranges the account names.
pub(crate) const TOP: usize = 10;

/// The `n` rows with the most CPU bytes, most first (ties by address).
pub(crate) fn top(mut rows: Vec<Row>, n: usize) -> Vec<Row> {
    rows.retain(|r| r.cpu_bytes() > 0);
    rows.sort_unstable_by(|a, b| b.cpu_bytes().cmp(&a.cpu_bytes()).then(a.start.cmp(&b.start)));
    rows.truncate(n);
    rows
}

/// The frame-rate line's own numbers over one account period, summed
/// from its one-second buckets (raw: microseconds and bytes).
#[derive(Default, Clone, Debug)]
pub(crate) struct Window {
    pub(crate) frames: u64,
    pub(crate) us: u64,
    /// Preparing submissions, waiting for the GPU.
    pub(crate) cpu_us: u64,
    pub(crate) gpu_us: u64,
    /// The frame line's "copies" (refreshes with their write-watch askings,
    /// our own copies and fills) and their bytes.
    pub(crate) copy_us: u64,
    pub(crate) copy_bytes: u64,
    /// Texture refreshes and the bytes they read.
    pub(crate) texture_us: u64,
    pub(crate) texture_bytes: u64,
    /// Copying back and the bytes written to title memory.
    pub(crate) back_us: u64,
    pub(crate) back_bytes: u64,
}

impl Window {
    /// Adds one second's buckets.
    pub(crate) fn add(&mut self, frames: u64, us: u64, submission: [u64; 2], preparing: [u64; 4], back: [u64; 2]) {
        self.frames += frames;
        self.us += us;
        self.cpu_us += submission[0];
        self.gpu_us += submission[1];
        self.copy_us += preparing[0];
        self.copy_bytes += preparing[1];
        self.texture_us += preparing[2];
        self.texture_bytes += preparing[3];
        self.back_us += back[0];
        self.back_bytes += back[1];
    }
}

/// What our layer and the process hold at the end of a period, in bytes
/// (page faults: a count).
#[derive(Default, Clone, Debug)]
pub(crate) struct Held {
    /// Host copies' PC sides: of title buffers, of windows (card copies'
    /// CPU sides included: they are PC memory).
    pub(crate) pc_buffers: u64,
    pub(crate) pc_windows: u64,
    /// Snapshots held by copies, and kept in the pool for reuse.
    pub(crate) snapshots: u64,
    pub(crate) snapshot_pool: u64,
    /// Texture staging buffers, and our threads' texture read buffers.
    pub(crate) texture_staging: u64,
    pub(crate) texture_read: u64,
    /// Upload memory held by host lists (scratch chunks, spare ones
    /// included, and other upload buffers).
    pub(crate) uploads: u64,
    /// Card copies' card sides, and their pieces' hashes, in the card's
    /// memory; the card's local memory used and its budget.
    pub(crate) card_copies: u64,
    pub(crate) card_hashes: u64,
    pub(crate) card_used: u64,
    pub(crate) card_budget: u64,
    /// The process: private bytes, working set; page faults (soft and hard
    /// together) in the period; the PC's free and total physical memory.
    pub(crate) private: u64,
    pub(crate) working_set: u64,
    pub(crate) faults: u64,
    pub(crate) ram_free: u64,
    pub(crate) ram_total: u64,
}

/// One account period.
#[derive(Clone, Debug)]
pub(crate) struct Period {
    pub(crate) window: Window,
    pub(crate) bytes: Vec<u64>,
    pub(crate) copy_only_us: u64,
    pub(crate) held: Held,
    pub(crate) top: Vec<Row>,
}

/// Refresh copies slower than this (bytes per second) over a period of at
/// least [`SLOW_MIN_US`] of them are flagged: title memory read at that
/// rate is coming from the page file, not RAM (copies run at gigabytes a
/// second when it is resident).
const SLOW_BYTES_PER_S: f64 = 200e6;
const SLOW_MIN_US: u64 = 100_000;

/// Whether copies of `bytes` that took `us` ran slow (paging?).
pub(crate) fn slow(bytes: u64, us: u64) -> bool {
    us >= SLOW_MIN_US && (bytes as f64) < SLOW_BYTES_PER_S * us as f64 / 1e6
}

fn rate_mb_s(bytes: u64, us: u64) -> f64 {
    if us == 0 { 0.0 } else { bytes as f64 / us as f64 }
}

impl Period {
    fn op(&self, op: Op) -> u64 {
        (0..MEMS * SIDES).map(|i| self.bytes[op as usize * MEMS * SIDES + i]).sum()
    }

    fn op_mem(&self, op: Op, mem: usize) -> u64 {
        (0..SIDES).map(|s| self.bytes[(op as usize * MEMS + mem) * SIDES + s]).sum()
    }

    fn op_side(&self, op: Op, side: usize) -> u64 {
        (0..MEMS).map(|m| self.bytes[(op as usize * MEMS + m) * SIDES + side]).sum()
    }

    fn refresh_bytes(&self) -> u64 {
        self.op(Op::Copied) + self.op(Op::Whole)
    }

    /// Whether the period's refresh copies ran slow ([`slow`]).
    pub(crate) fn slow(&self) -> bool {
        slow(self.refresh_bytes(), self.copy_only_us)
    }

    fn frames(&self) -> f64 {
        self.window.frames.max(1) as f64
    }

    /// The bytes line: per frame, the frame line's times with their bytes
    /// and rates, then each operation's MB with its kinds.
    pub(crate) fn bytes_line(&self) -> String {
        use std::fmt::Write;
        let w = &self.window;
        let f = self.frames();
        let ms = |us: u64| us as f64 / 1e3 / f;
        let mb = |b: u64| b as f64 / 1e6 / f;
        let mut text = format!(
            "account over {:.1} s, {} frames, per frame: preparing {:.1} ms, GPU {:.1} ms; copies {:.1} ms {:.1} MB ({:.0} MB/s; \
             the refresh copies alone {:.1} ms for {:.1} MB, {:.0} MB/s), textures {:.1} ms {:.1} MB ({:.0} MB/s), copy back {:.1} ms \
             {:.1} MB",
            w.us as f64 / 1e6,
            w.frames,
            ms(w.cpu_us),
            ms(w.gpu_us),
            ms(w.copy_us),
            mb(w.copy_bytes),
            rate_mb_s(w.copy_bytes, w.copy_us),
            ms(self.copy_only_us),
            mb(self.refresh_bytes()),
            rate_mb_s(self.refresh_bytes(), self.copy_only_us),
            ms(w.texture_us),
            mb(w.texture_bytes),
            rate_mb_s(w.texture_bytes, w.texture_us),
            ms(w.back_us),
            mb(w.back_bytes),
        );
        if self.slow() {
            let _ = write!(
                text,
                "; SLOW (paging?): refresh copies at {:.0} MB/s with {} page faults in the period ({:.0}/s)",
                rate_mb_s(self.refresh_bytes(), self.copy_only_us),
                self.held.faults,
                self.held.faults as f64 / (w.us.max(1) as f64 / 1e6)
            );
        }
        let _ = write!(text, "; MB per frame by operation [by memory | by side]:");
        for op in OP_ALL {
            let total = self.op(op);
            if total == 0 {
                continue;
            }
            let _ = write!(text, " {} {:.1} [", op.name(), mb(total));
            let mut first = true;
            for (m, name) in MEM_NAMES.iter().enumerate() {
                let v = self.op_mem(op, m);
                if v != 0 {
                    let _ = write!(text, "{}{name} {:.1}", if first { "" } else { ", " }, mb(v));
                    first = false;
                }
            }
            let _ = write!(text, " |");
            first = true;
            for (s, name) in SIDE_NAMES.iter().enumerate() {
                let v = self.op_side(op, s);
                if v != 0 {
                    let _ = write!(text, "{}{name} {:.1}", if first { " " } else { ", " }, mb(v));
                    first = false;
                }
            }
            let _ = write!(text, "];");
        }
        text
    }

    /// The memory line: what our layer and the process hold now.
    pub(crate) fn memory_line(&self) -> String {
        let h = &self.held;
        let mb = |b: u64| b >> 20;
        let secs = (self.window.us.max(1) as f64) / 1e6;
        format!(
            "account memory now (MiB): host copies on the PC {} (buffers {}, windows {}), snapshots {} held + {} pooled, texture \
             staging {} + read buffers {}, list upload memory {}, card copies {} + hashes {} (the card's memory {} used of a budget \
             of {}); the process: private {}, working set {}, page faults {:.0}/s (soft and hard: hard alone is not counted), the \
             PC's memory free {} of {}",
            mb(h.pc_buffers + h.pc_windows),
            mb(h.pc_buffers),
            mb(h.pc_windows),
            mb(h.snapshots),
            mb(h.snapshot_pool),
            mb(h.texture_staging),
            mb(h.texture_read),
            mb(h.uploads),
            mb(h.card_copies),
            mb(h.card_hashes),
            mb(h.card_used),
            mb(h.card_budget),
            mb(h.private),
            mb(h.working_set),
            h.faults as f64 / secs,
            mb(h.ram_free),
            mb(h.ram_total),
        )
    }

    /// The top ranges line: the copies with the most CPU bytes over the
    /// period (MB in all, not per frame), with the frames the CPU, the GPU
    /// and our merge wrote them.
    pub(crate) fn top_line(&self) -> String {
        use std::fmt::Write;
        let mut text = format!(
            "account: the {} host copies with the most CPU bytes in {} frames (MB in the period: up copied, cmp compared, diff in \
             differing pieces, back merged, card sent up; frames written by the cpu, declared written by the gpu, merged into):",
            self.top.len(),
            self.window.frames
        );
        for r in &self.top {
            let mb = |b: u64| b as f64 / 1e6;
            let _ = write!(
                text,
                " {:#x}+{:.1}MB {} {}: up {:.1} cmp {:.1} diff {:.1} back {:.1} card {:.1}, cpu {} gpu {} merge {};",
                r.start,
                mb(r.len),
                SIDE_NAMES[r.side as usize],
                MEM_NAMES[r.mem as usize],
                mb(r.copied),
                mb(r.compared),
                mb(r.differing),
                mb(r.merged),
                mb(r.sent),
                r.cpu_frames,
                r.gpu_frames,
                r.merge_frames
            );
        }
        text
    }
}

/// The columns before the operations'.
const CSV_LEAD: [&str; 33] = [
    "unix_ms",
    "seconds",
    "frames",
    "ms_per_frame",
    "preparing_ms",
    "gpu_ms",
    "copies_ms",
    "copies_mb",
    "copies_mb_s",
    "refresh_copy_ms",
    "refresh_copy_mb",
    "refresh_copy_mb_s",
    "textures_ms",
    "textures_mb",
    "copy_back_ms",
    "copy_back_mb",
    "slow",
    "page_faults",
    "page_faults_per_s",
    "private_mib",
    "working_set_mib",
    "ram_free_mib",
    "pc_buffers_mib",
    "pc_windows_mib",
    "snapshots_mib",
    "snapshot_pool_mib",
    "texture_staging_mib",
    "texture_read_mib",
    "upload_mib",
    "card_copies_mib",
    "card_hashes_mib",
    "card_used_mib",
    "card_budget_mib",
];

/// The columns of each top range.
const CSV_RANGE: [&str; 12] = [
    "start", "len", "side", "mem", "up_mb", "cmp_mb", "diff_mb", "back_mb", "card_mb", "cpu_frames", "gpu_frames", "merge_frames",
];

/// The CSV header: the lead columns; per operation its MB per frame in
/// all, by kind of memory and by side; then the top ranges' columns.
pub(crate) fn csv_header() -> String {
    let mut cols: Vec<String> = CSV_LEAD.iter().map(|c| c.to_string()).collect();
    for op in OP_ALL {
        cols.push(format!("{}_mb", op.name()));
        cols.extend(MEM_NAMES.iter().map(|m| format!("{}_{m}_mb", op.name())));
        cols.extend(SIDE_NAMES.iter().map(|s| format!("{}_{s}_mb", op.name())));
    }
    for i in 1..=TOP {
        cols.extend(CSV_RANGE.iter().map(|c| format!("top{i}_{c}")));
    }
    cols.join(",")
}

/// The period's CSV row, `unix_ms` its end; as many fields as
/// [`csv_header`] has columns (ranges past the top's are empty).
pub(crate) fn csv_row(p: &Period, unix_ms: u128) -> String {
    let w = &p.window;
    let h = &p.held;
    let f = p.frames();
    let ms = |us: u64| format!("{:.2}", us as f64 / 1e3 / f);
    let mb = |b: u64| format!("{:.3}", b as f64 / 1e6 / f);
    let mib = |b: u64| (b >> 20).to_string();
    let secs = w.us.max(1) as f64 / 1e6;
    let mut cols: Vec<String> = vec![
        unix_ms.to_string(),
        format!("{:.2}", w.us as f64 / 1e6),
        w.frames.to_string(),
        format!("{:.1}", w.us as f64 / 1e3 / f),
        ms(w.cpu_us),
        ms(w.gpu_us),
        ms(w.copy_us),
        mb(w.copy_bytes),
        format!("{:.0}", rate_mb_s(w.copy_bytes, w.copy_us)),
        ms(p.copy_only_us),
        mb(p.refresh_bytes()),
        format!("{:.0}", rate_mb_s(p.refresh_bytes(), p.copy_only_us)),
        ms(w.texture_us),
        mb(w.texture_bytes),
        ms(w.back_us),
        mb(w.back_bytes),
        u8::from(p.slow()).to_string(),
        h.faults.to_string(),
        format!("{:.0}", h.faults as f64 / secs),
        mib(h.private),
        mib(h.working_set),
        mib(h.ram_free),
        mib(h.pc_buffers),
        mib(h.pc_windows),
        mib(h.snapshots),
        mib(h.snapshot_pool),
        mib(h.texture_staging),
        mib(h.texture_read),
        mib(h.uploads),
        mib(h.card_copies),
        mib(h.card_hashes),
        mib(h.card_used),
        mib(h.card_budget),
    ];
    for op in OP_ALL {
        cols.push(mb(p.op(op)));
        cols.extend((0..MEMS).map(|m| mb(p.op_mem(op, m))));
        cols.extend((0..SIDES).map(|s| mb(p.op_side(op, s))));
    }
    for i in 0..TOP {
        match p.top.get(i) {
            Some(r) => {
                let total = |b: u64| format!("{:.3}", b as f64 / 1e6);
                cols.extend([
                    format!("{:#x}", r.start),
                    r.len.to_string(),
                    SIDE_NAMES[r.side as usize].to_string(),
                    MEM_NAMES[r.mem as usize].to_string(),
                    total(r.copied),
                    total(r.compared),
                    total(r.differing),
                    total(r.merged),
                    total(r.sent),
                    r.cpu_frames.to_string(),
                    r.gpu_frames.to_string(),
                    r.merge_frames.to_string(),
                ]);
            }
            None => cols.extend(std::iter::repeat_n(String::new(), CSV_RANGE.len())),
        }
    }
    cols.join(",")
}

/// The process's private bytes, working set and page fault count (soft
/// and hard: `GetProcessMemoryInfo` does not tell them apart).
fn process_memory() -> (u64, u64, u32) {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS_EX {
        cb: size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a structure of the size given (the
    // extended form, which the call fills when `cb` says so).
    let ok = unsafe {
        K32GetProcessMemoryInfo(GetCurrentProcess(), (&raw mut c).cast::<PROCESS_MEMORY_COUNTERS>(), c.cb)
    } != 0;
    if !ok {
        return (0, 0, 0);
    }
    (c.PrivateUsage as u64, c.WorkingSetSize as u64, c.PageFaultCount)
}

/// The PC's free and total physical memory in bytes.
fn system_memory() -> (u64, u64) {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 {
        return (0, 0);
    }
    (m.ullAvailPhys, m.ullTotalPhys)
}

/// The page fault count at the last account (the counter is 32 bits and
/// wraps; the difference is taken modulo that).
static LAST_FAULTS: Mutex<Option<u32>> = Mutex::new(None);

/// Takes the counters into a period over `window`.
fn gather(window: Window) -> Period {
    let bytes: Vec<u64> = BYTES.iter().map(|b| b.swap(0, Relaxed)).collect();
    let copy_only_us = COPY_US.swap(0, Relaxed);
    let walk = crate::shadow::account_walk();
    let (private, working_set, faults_now) = process_memory();
    let faults = {
        let mut last = LAST_FAULTS.lock().unwrap_or_else(|e| e.into_inner());
        let delta = last.map_or(0, |was| faults_now.wrapping_sub(was));
        *last = Some(faults_now);
        u64::from(delta)
    };
    let (ram_free, ram_total) = system_memory();
    let [card_budget, card_used, ..] = crate::host::host().and_then(|h| h.video_memory()).unwrap_or_default();
    let alive = &crate::host::ALIVE;
    let held = Held {
        pc_buffers: walk.pc_buffers,
        pc_windows: walk.pc_windows,
        snapshots: SNAPSHOT_HELD.load(Relaxed),
        snapshot_pool: walk.snapshot_pool,
        texture_staging: crate::texture::staged_bytes(),
        texture_read: TEXTURE_READ_HELD.load(Relaxed),
        uploads: alive[2].load(Relaxed) + alive[7].load(Relaxed),
        card_copies: walk.card_copies,
        card_hashes: walk.card_hashes,
        card_used,
        card_budget,
        private,
        working_set,
        faults,
        ram_free,
        ram_total,
    };
    Period { window, bytes, copy_only_us, held, top: top(walk.rows, TOP) }
}

/// What [`crate::shadow::account_walk`] finds.
#[derive(Default)]
pub(crate) struct Walk {
    pub(crate) rows: Vec<Row>,
    pub(crate) pc_buffers: u64,
    pub(crate) pc_windows: u64,
    pub(crate) snapshot_pool: u64,
    pub(crate) card_copies: u64,
    pub(crate) card_hashes: u64,
}

/// `VICTORY_ACCOUNT_CSV`: the file the account appends its rows to, once
/// opened (`None`: not asked for, or it could not be opened).
fn csv_file() -> Option<&'static Mutex<std::fs::File>> {
    static FILE: std::sync::OnceLock<Option<Mutex<std::fs::File>>> = std::sync::OnceLock::new();
    FILE.get_or_init(|| {
        let path = std::env::var_os("VICTORY_ACCOUNT_CSV").filter(|p| !p.is_empty())?;
        match std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            Ok(file) => Some(Mutex::new(file)),
            Err(e) => {
                tracing::warn!(error = %e, path = ?path, "the account's CSV file could not be opened (VICTORY_ACCOUNT_CSV)");
                None
            }
        }
    })
    .as_ref()
}

/// Logs the account of the period `window` ends, and appends its CSV row
/// if asked to. Called from the frame-rate line's five-second block.
pub(crate) fn log(window: Window) {
    let period = gather(window);
    tracing::info!("{}", period.bytes_line());
    tracing::info!("{}", period.memory_line());
    tracing::info!("{}", period.top_line());
    if let Some(file) = csv_file() {
        use std::io::Write;
        let mut file = file.lock().unwrap_or_else(|e| e.into_inner());
        let empty = file.metadata().map(|m| m.len() == 0).unwrap_or(false);
        let unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        let mut text = String::new();
        if empty {
            text.push_str(&csv_header());
            text.push('\n');
        }
        text.push_str(&csv_row(&period, unix_ms));
        text.push('\n');
        if let Err(e) = file.write_all(text.as_bytes()) {
            tracing::warn!(error = %e, "the account's CSV row could not be written");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(start: u64, side: Side, copied: u64, compared: u64, merged: u64) -> Row {
        Row {
            start,
            len: 1 << 20,
            side,
            mem: Mem::Watched,
            copied,
            compared,
            differing: 0,
            merged,
            sent: 0,
            cpu_frames: 0,
            gpu_frames: 0,
            merge_frames: 0,
        }
    }

    fn period(cells: &[(Op, Mem, Side, u64)]) -> Period {
        let mut bytes = vec![0u64; CELLS];
        for &(op, mem, side, b) in cells {
            bytes[cell(op, mem, side)] += b;
        }
        Period {
            window: Window { frames: 10, us: 5_000_000, ..Window::default() },
            bytes,
            copy_only_us: 0,
            held: Held::default(),
            top: Vec::new(),
        }
    }

    /// Every cell is its own, and the sums by operation, kind of memory
    /// and side each count every cell of the operation once.
    #[test]
    fn cells_are_distinct_and_sums_count_each_once() {
        let mut seen = std::collections::HashSet::new();
        for op in OP_ALL {
            for m in 0..MEMS {
                for s in 0..SIDES {
                    let c = cell(op, Mem::from_u8(m as u8), Side::from_index(s));
                    assert!(c < CELLS && seen.insert(c));
                }
            }
        }
        assert_eq!(seen.len(), CELLS);
        let p = period(&[
            (Op::Copied, Mem::View, Side::PcBuffer, 100),
            (Op::Copied, Mem::View, Side::CardBuffer, 20),
            (Op::Copied, Mem::Watched, Side::PcWindow, 3),
            (Op::Merged, Mem::View, Side::PcBuffer, 7),
        ]);
        assert_eq!(p.op(Op::Copied), 123);
        assert_eq!(p.op_mem(Op::Copied, Mem::View as usize), 120);
        assert_eq!(p.op_mem(Op::Copied, Mem::Watched as usize), 3);
        assert_eq!(p.op_side(Op::Copied, Side::PcBuffer as usize), 100);
        assert_eq!(p.op_side(Op::Copied, Side::CardBuffer as usize), 20);
        assert_eq!(p.op(Op::Merged), 7);
        assert_eq!(p.op(Op::Compared), 0);
        let by_mem: u64 = (0..MEMS).map(|m| p.op_mem(Op::Copied, m)).sum();
        let by_side: u64 = (0..SIDES).map(|s| p.op_side(Op::Copied, s)).sum();
        assert_eq!((by_mem, by_side), (123, 123));
    }

    /// A tally sums its bytes into its fields and counts a frame once
    /// however often it is marked in it; a take empties it.
    #[test]
    fn a_tally_counts_frames_once_and_empties_on_take() {
        let t = Tally::default();
        t.add(Op::Copied, 10);
        t.add(Op::Whole, 5);
        t.add(Op::Under, 1);
        t.add(Op::Compared, 100);
        t.add(Op::Merged, 2);
        t.add(Op::TexRead, 999);
        t.set_mem(Mem::View);
        for _ in 0..3 {
            t.mark(Wrote::Gpu);
        }
        t.mark(Wrote::Cpu);
        // `FRAME` is shared with other tests, which do not present: the
        // marks above fall in one frame, or at most two.
        let r = t.take(0x1000, 64, Side::PcWindow).expect("a row");
        assert_eq!((r.copied, r.compared, r.merged, r.differing, r.sent), (16, 100, 2, 0, 0));
        assert_eq!(r.mem, Mem::View);
        assert!((1..=2).contains(&r.gpu_frames));
        assert_eq!((r.cpu_frames, r.merge_frames), (1, 0));
        assert_eq!(t.take(0x1000, 64, Side::PcWindow), None);
        // Marked again in the same frame: not counted again.
        let before = FRAME.load(Relaxed);
        t.mark(Wrote::Gpu);
        if FRAME.load(Relaxed) == before {
            assert_eq!(t.take(0x1000, 64, Side::PcWindow), None);
        }
    }

    /// The top keeps the rows with the most CPU bytes, most first, leaves
    /// out rows without any, and does not count the card's own compares.
    #[test]
    fn the_top_ranks_by_cpu_bytes() {
        let rows = vec![
            row(0x10, Side::PcBuffer, 5, 0, 0),
            row(0x20, Side::CardBuffer, 1, 1_000, 0),
            row(0x30, Side::PcWindow, 0, 50, 1),
            row(0x40, Side::PcBuffer, 0, 0, 0),
            row(0x50, Side::PcBuffer, 0, 0, 5),
        ];
        let t = top(rows.clone(), 10);
        let starts: Vec<u64> = t.iter().map(|r| r.start).collect();
        // 0x30: 51; 0x10 and 0x50: 5 each, by address; 0x20: 1 (its
        // compare is the card's); 0x40: nothing.
        assert_eq!(starts, vec![0x30, 0x10, 0x50, 0x20]);
        assert_eq!(top(rows, 2).len(), 2);
        assert!(top(Vec::new(), 10).is_empty());
    }

    /// Slow means under 200 MB/s over at least a tenth of a second of
    /// copies; a short period is never flagged.
    #[test]
    fn slow_periods_are_flagged() {
        assert!(slow(10_000_000, 100_000)); // 100 MB/s over 0.1 s
        assert!(!slow(30_000_000, 100_000)); // 300 MB/s
        assert!(!slow(1_000, 99_999)); // too short to tell
        assert!(slow(0, 1_000_000));
        assert!(!slow(200_000_000, 1_000_000)); // exactly the rate
        let mut p = period(&[(Op::Copied, Mem::View, Side::PcBuffer, 6_700_000)]);
        p.copy_only_us = 1_000_000;
        p.held.faults = 1234;
        assert!(p.slow());
        assert!(p.bytes_line().contains("SLOW (paging?)"));
        assert!(p.bytes_line().contains("1234 page faults"));
        p.copy_only_us = 10_000;
        assert!(!p.slow());
        assert!(!p.bytes_line().contains("SLOW"));
    }

    /// Every CSV row has as many fields as the header, with ranges or
    /// without, and the header names are unique.
    #[test]
    fn csv_rows_match_the_header() {
        let header = csv_header();
        let names: Vec<&str> = header.split(',').collect();
        let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
        assert_eq!(unique.len(), names.len());
        assert_eq!(names.len(), CSV_LEAD.len() + OPS * (1 + MEMS + SIDES) + TOP * CSV_RANGE.len());
        let mut p = period(&[(Op::Copied, Mem::View, Side::PcBuffer, 10_000_000), (Op::Compared, Mem::Watched, Side::PcWindow, 1)]);
        let empty = csv_row(&p, 42);
        assert_eq!(empty.split(',').count(), names.len());
        assert!(empty.starts_with("42,5.00,10,500.0,"));
        p.top = vec![row(0xd_1200_0000, Side::PcWindow, 3_000_000, 0, 0)];
        let full = csv_row(&p, 43);
        let fields: Vec<&str> = full.split(',').collect();
        assert_eq!(fields.len(), names.len());
        let at = names.iter().position(|&n| n == "top1_start").expect("column");
        assert_eq!(&fields[at..at + 5], &["0xd12000000", "1048576", "pc_window", "watched", "3.000"]);
        let copied = names.iter().position(|&n| n == "copied_mb").expect("column");
        // 10 MB over 10 frames.
        assert_eq!(fields[copied], "1.000");
        assert_eq!(fields[names.iter().position(|&n| n == "copied_view_mb").expect("column")], "1.000");
        assert_eq!(fields[names.iter().position(|&n| n == "copied_pc_buffer_mb").expect("column")], "1.000");
        assert_eq!(fields[names.iter().position(|&n| n == "top2_start").expect("column")], "");
    }

    /// Refreshes on several threads: the copies keep their share of the
    /// time the refreshes took.
    #[test]
    fn parallel_copy_time_is_its_share_of_the_wall_time() {
        // 8 threads, 800 ms of refreshes in 100 ms, 400 ms of it copying.
        assert_eq!(wall_share(400_000, 800_000, 100_000), 50_000);
        // One thread: as measured.
        assert_eq!(wall_share(40_000, 80_000, 90_000), 40_000);
        assert_eq!(wall_share(0, 800_000, 100_000), 0);
    }

    /// The window sums its seconds' buckets field by field.
    #[test]
    fn the_window_sums_its_seconds() {
        let mut w = Window::default();
        w.add(3, 1_000_000, [10, 20], [1, 2, 3, 4], [5, 6]);
        w.add(4, 1_100_000, [1, 1], [1, 1, 1, 1], [1, 1]);
        assert_eq!((w.frames, w.us, w.cpu_us, w.gpu_us), (7, 2_100_000, 11, 21));
        assert_eq!((w.copy_us, w.copy_bytes, w.texture_us, w.texture_bytes), (2, 3, 4, 5));
        assert_eq!((w.back_us, w.back_bytes), (6, 7));
    }

    /// The bytes line names only the operations and kinds that did
    /// something.
    #[test]
    fn the_bytes_line_names_what_happened() {
        let p = period(&[
            (Op::Copied, Mem::View, Side::PcBuffer, 20_000_000),
            (Op::Copied, Mem::Watched, Side::CardBuffer, 10_000_000),
        ]);
        let line = p.bytes_line();
        assert!(line.contains("copied 3.0 [watched 1.0, view 2.0 | pc_buffer 2.0, card_buffer 1.0];"), "{line}");
        assert!(!line.contains("merged"));
        assert!(!line.contains("SLOW"));
    }
}
