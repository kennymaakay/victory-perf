//! Instructions the console's CPU (Zen 2) has and an Intel PC's lacks, emulated: the title's
//! code raises STATUS_ILLEGAL_INSTRUCTION on them (the title zeroes memory with a `clzero` loop at
//! its first frames), and the vectored handler steps over them here.
//!
//! Ported from XCL (github.com/ivall/xcl, GPL-2.0-only): `insn.rs` (decoder, as
//! `console_insn_decode.rs`, unchanged) and `eh.rs` `emulate_console_only`.

#[path = "console_insn_decode.rs"]
mod decode;

use decode::ConsoleOnly;
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use windows_sys::Win32::System::Diagnostics::Debug::CONTEXT;

const CONTINUE_EXECUTION: i32 = -1;

/// Whether what Windows said about memory is remembered per thread (the regions it found
/// writable, the code bytes it found readable) and not asked again at every emulated
/// instruction. A query walks the pages of the region it answers for: asked from outside, it
/// took 2-9 ms in the title's heaps of 0.6-2.2 GB and 89 ms in its 6.2 GB page pool (run 30),
/// and the title zeroes blocks of them with `clzero` loops all through a frame: six of its
/// threads spent 25-78 % of their time inside that query (runs 29 and 30), about 27 s of
/// processor time in Windows per 9 s frame. On by default; `VICTORY_CLZERO_ASK=1` asks every
/// time, as before.
fn remembers() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CLZERO_ASK").is_none_or(|v| v != "1"))
}

/// Answers a remembered region gives before Windows is asked about it again: one freed or
/// protected meanwhile is forgotten then.
const ANSWERS: u32 = 4096;

thread_local! {
    /// The regions Windows last said are committed and writable (start, end), newest first,
    /// and the answers they gave since they were emptied.
    static WRITABLE: Cell<([(usize, usize); 4], u32)> = const { Cell::new(([(0, 0); 4], 0)) };
    /// Faulting addresses and the code bytes found readable there, newest first.
    static CODE: Cell<[(usize, usize); 8]> = const { Cell::new([(0, 0); 8]) };
}

/// Instructions emulated, the times Windows was asked about memory for them, and the bytes
/// `clzero` loops zeroed, since the start.
static EMULATED: AtomicU64 = AtomicU64::new(0);
static ASKED: AtomicU64 = AtomicU64::new(0);
static ZEROED: AtomicU64 = AtomicU64::new(0);

/// Instructions emulated since the start, by kind ([`kind`] gives the index, [`KIND_NAMES`]
/// the names). Each one was an exception, so how many of each a second the title runs is what
/// this emulation costs it. `monitorx`, `mwaitx` and `clwb` are patched in place after their
/// first trap ([`patch`]), so their counts fall to about nothing once the title's wait loops
/// have each run once; the rest trap every time.
static KINDS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];
const KIND_NAMES: [&str; 10] = ["monitorx", "mwaitx", "clzero", "clzero loops", "clwb", "rdpru", "rdpid", "extrq", "insertq", "other"];

fn kind(what: &ConsoleOnly) -> usize {
    match what {
        ConsoleOnly::Monitorx => 0,
        ConsoleOnly::Mwaitx => 1,
        ConsoleOnly::Clzero => 2,
        ConsoleOnly::ClzeroLoop { .. } => 3,
        ConsoleOnly::Clwb => 4,
        ConsoleOnly::Rdpru => 5,
        ConsoleOnly::Rdpid(_) => 6,
        ConsoleOnly::Extrq { .. } => 7,
        ConsoleOnly::Insertq { .. } => 8,
        ConsoleOnly::Ud2 => 9,
    }
}

/// One line every five seconds while the title runs: the console instructions emulated per
/// second, by kind, and the processor time of the whole process in that window, user and
/// kernel (in cores: 1.0 is one logical processor kept busy). Each emulation is a trip into
/// Windows, so a high kernel share with many `mwaitx` a second says the traps cost the title
/// real time. `VICTORY_INSN_STATS=0` turns the line off.
pub(crate) fn start_stats() {
    if std::env::var_os("VICTORY_INSN_STATS").is_some_and(|v| v == "0") {
        return;
    }
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("victory-insn-stats".into()).spawn(|| {
            const PERIOD: std::time::Duration = std::time::Duration::from_secs(5);
            let mut last = [0u64; 10];
            let mut last_patched = 0;
            let mut times = process_times();
            loop {
                std::thread::sleep(PERIOD);
                let now: [u64; 10] = std::array::from_fn(|i| KINDS[i].load(Relaxed));
                let patched = PATCHED.load(Relaxed);
                let next = process_times();
                let total: u64 = now.iter().zip(&last).map(|(n, l)| n - l).sum();
                if total > 0 || patched != last_patched {
                    let secs = PERIOD.as_secs_f64();
                    let per: Vec<String> = now
                        .iter()
                        .zip(&last)
                        .zip(KIND_NAMES)
                        .filter(|((n, l), _)| **n > **l)
                        .map(|((n, l), name)| format!("{name} {:.0}", (n - l) as f64 / secs))
                        .collect();
                    let (user, kernel) = match (times, next) {
                        (Some((u0, k0)), Some((u1, k1))) => {
                            // 100 ns units over the window: cores kept busy
                            let cores = |a: u64, b: u64| (b.saturating_sub(a)) as f64 / (secs * 1e7);
                            (cores(u0, u1), cores(k0, k1))
                        }
                        _ => (f64::NAN, f64::NAN),
                    };
                    tracing::info!(
                        "console CPU instructions emulated per second (each one an exception): {}; instructions patched in place so far: {patched}; the process's processor time: {user:.2} cores in user mode, {kernel:.2} in the kernel",
                        if per.is_empty() { "none".to_string() } else { per.join(", ") }
                    );
                }
                last = now;
                last_patched = patched;
                times = next;
            }
        });
    });
}

/// The process's user and kernel processor time so far, in 100 ns units.
fn process_times() -> Option<(u64, u64)> {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: the pseudo handle of this process and valid out pointers.
    let ok = unsafe { GetProcessTimes(GetCurrentProcess(), &mut created, &mut exited, &mut kernel, &mut user) } != 0;
    let t = |f: FILETIME| (u64::from(f.dwHighDateTime) << 32) | u64::from(f.dwLowDateTime);
    ok.then(|| (t(user), t(kernel)))
}

fn readable(va: usize, bytes: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let found = unsafe { VirtualQuery(va as *const _, &mut m, size_of_val(&m)) } != 0;
    ASKED.fetch_add(1, Relaxed);
    found && m.State == MEM_COMMIT && m.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0 && m.BaseAddress as usize + m.RegionSize >= va + bytes
}

fn writable(va: usize, bytes: usize) -> bool {
    writable_to(va, va + bytes) == Some(va + bytes)
}

/// Where the committed, writable memory that `va` lies in ends, at most `end`; None if `va`
/// is not in such memory. A region found is remembered ([`remembers`]).
/// shortcut: a remembered region that was freed or protected since is written all the same
/// for up to [`ANSWERS`] answers: the write then faults inside the handler, which the guard
/// on view memory opens as it does any write, and which is a crash otherwise (as the
/// unchecked single-line emulation behind this one already is).
fn writable_to(va: usize, end: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READWRITE, PAGE_READWRITE, VirtualQuery};
    let remember = remembers();
    if remember {
        let (regions, answers) = WRITABLE.with(Cell::get);
        if let Some(&(_, to)) = regions.iter().find(|&&(from, to)| from <= va && va < to) {
            let next = if answers >= ANSWERS { ([(0, 0); 4], 0) } else { (regions, answers + 1) };
            WRITABLE.with(|w| w.set(next));
            return Some(to.min(end));
        }
    }
    // SAFETY: plain data filled in by the query.
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let found = unsafe { VirtualQuery(va as *const _, &mut m, size_of_val(&m)) } != 0;
    ASKED.fetch_add(1, Relaxed);
    if !found || m.State != MEM_COMMIT || m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) == 0 {
        return None;
    }
    let (from, to) = (m.BaseAddress as usize, m.BaseAddress as usize + m.RegionSize);
    if remember {
        WRITABLE.with(|w| {
            let (mut regions, answers) = w.get();
            regions.rotate_right(1);
            regions[0] = (from, to);
            w.set((regions, answers));
        });
    }
    Some(to.min(end))
}

/// Whether `monitorx` and `mwaitx` are patched in place at their first trap ([`patch`]). The
/// title's wait loops ran about 100-130 thousand of each a second on an Intel i9-9900, each an
/// exception, and the process spent 0.35-0.45 of a core in the kernel on them. On by default;
/// `VICTORY_PATCH_WAITS=0` emulates every one, as before (one exception and one spin hint per
/// `mwaitx`), to compare against: a patched `mwaitx` is a plain NOP and its loop spins hot (see
/// [`replacement`]). `clwb` is patched either way, as it always was.
fn patches_waits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PATCH_WAITS").is_none_or(|v| v != "0"))
}

/// Instructions patched in place since the start.
static PATCHED: AtomicU64 = AtomicU64::new(0);

/// An instruction's bytes (the first `.1` of them).
type Code = ([u8; 16], usize);

/// Every address a patch was tried at, with the whole instruction as the patch left it (None:
/// it was refused and the instruction stays emulated). The lock also keeps patches one at a
/// time: two threads patching in the same page would otherwise restore its protection under
/// each other, and the second one's write would fault.
static SITES: std::sync::Mutex<std::collections::BTreeMap<usize, Option<Code>>> = std::sync::Mutex::new(std::collections::BTreeMap::new());

fn sites() -> std::sync::MutexGuard<'static, std::collections::BTreeMap<usize, Option<Code>>> {
    SITES.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// One byte of an instruction rewritten: the byte at `offset` in it goes from `old` to `new`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Rewrite {
    offset: usize,
    old: u8,
    new: u8,
}

impl Rewrite {
    /// The instruction `code` as the rewrite leaves it.
    fn apply(self, code: &[u8]) -> Code {
        let mut new = [0u8; 16];
        new[..code.len()].copy_from_slice(code);
        new[self.offset] = self.new;
        (new, code.len())
    }
}

/// The one byte that changes in a console instruction (`code`, all of it) so that it never
/// traps again, or None when it stays emulated. It is always the opcode byte after `0F`,
/// turned into `1F`: `0F 1F /r` is the hint-NOP opcode (NOP Ev), a NOP on Intel and AMD whatever
/// its ModRM, which reads no memory, and the old and the new instruction have the same length
/// and the same bytes but that one, so whichever of the two a thread sees is valid code
/// ([`patch`] leans on that).
///
/// - `monitorx` / `mwaitx`, `0F 01 FA` / `0F 01 FB`: the `01` becomes `1F`, so `0F 1F FA` /
///   `0F 1F FB` (`nop edx` / `nop ebx`, ModRM register forms), three bytes as before. Only
///   the prefixes that do not change what the opcode is (REX, the address size, a segment)
///   are patched: with 66 / F2 / F3 in front it may be another instruction on the console, and
///   a guess is better emulated than written into the code.
/// - `clwb`, `66 [REX] 0F AE /6` and its memory operand: the `AE` becomes `1F`, so
///   `66 [REX] 0F 1F /6` with the same ModRM, SIB and displacement, a NOP Ew of the same length
///   whatever the operand (a hint with nothing to do on the PC, as the emulation always
///   treated it). Address-size and segment prefixes are patched as well; with F2 / F3 in front
///   it stays emulated (stepped over at every trap).
///
/// What the patched waits do: `mwaitx` waits until the line `monitorx` armed is written, a
/// timeout passes or anything else (an interrupt) ends it, and may return at once for no reason
/// at all, so a wait loop always checks its condition again after it; the title's do. A NOP is
/// such a return at once, and the emulation also only stepped over it, with one spin hint
/// (`pause`) a trap, so the loop works the same. What differs is the waiting itself: on the
/// console the core sleeps until the line changes; patched, the loop has no `pause` at all and
/// spins hot, a turn in some nanoseconds instead of the microsecond or more a trap took, polling
/// the line all the time and taking issue slots from the other hyperthread of its core, and a
/// loop that spins a set number of turns before it falls back to a wait in the kernel gets
/// there much sooner. Our earlier Leonida runtime ran these waits as the same NOPs on this
/// CPU. `pause` (`F3 90`) would need more than one byte changed, which no single store does
/// safely ([`patch`]); `VICTORY_PATCH_WAITS=0` keeps the emulation, one spin hint per trap, to
/// compare against. `monitorx` only arms the monitor, so with nothing waiting on it it does
/// nothing and a NOP is all it ever was here.
fn replacement(what: ConsoleOnly, code: &[u8]) -> Option<Rewrite> {
    let plain = |b: &u8| matches!(b, 0x40..=0x4F | 0x67 | 0x2E | 0x3E | 0x26 | 0x36 | 0x64 | 0x65);
    // where the 0F is, and the opcode byte after it that the patch rewrites
    let (escape, opcode) = match what {
        ConsoleOnly::Monitorx | ConsoleOnly::Mwaitx => {
            let escape = code.len().checked_sub(3)?;
            if !code[..escape].iter().all(plain) {
                return None;
            }
            (escape, 0x01)
        }
        ConsoleOnly::Clwb => {
            // prefixes are never 0F: the first one is the opcode's
            let escape = code.iter().position(|&b| b == 0x0F)?;
            let prefixes = &code[..escape];
            if !prefixes.contains(&0x66) || !prefixes.iter().all(|b| *b == 0x66 || plain(b)) {
                return None;
            }
            (escape, 0xAE)
        }
        _ => return None,
    };
    let offset = escape + 1;
    (code.get(escape) == Some(&0x0F) && code.get(offset) == Some(&opcode)).then_some(Rewrite { offset, old: opcode, new: 0x1F })
}

/// What a patch did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Patch {
    /// The new byte was written.
    Written,
    /// It was there already (another thread wrote it first).
    There,
    /// The byte there is neither the old nor the new one: nothing written.
    Changed,
    /// Windows refused to make the page writable: nothing written.
    Refused,
}

/// Writes `new` over `old` at `at`: one byte, with a compare-and-swap (`lock cmpxchg`), so
/// only if the byte is still the one decoded.
///
/// # Safety
/// `at` is writable, and nothing but other threads' handlers reads it meanwhile (their plain
/// reads race this store, which Rust's model does not allow but a byte store on x86 makes
/// harmless: they see the old byte or the new one).
unsafe fn patch_byte(at: *mut u8, old: u8, new: u8) -> Patch {
    use std::sync::atomic::{AtomicU8, Ordering::SeqCst};
    // SAFETY: a writable byte (the caller's promise), only ever written atomically here.
    let cell = unsafe { AtomicU8::from_ptr(at) };
    match cell.compare_exchange(old, new, SeqCst, SeqCst) {
        Ok(_) => Patch::Written,
        Err(now) if now == new => Patch::There,
        Err(_) => Patch::Changed,
    }
}

/// [`patch_byte`] on the process's code: the byte's page made writable for the store (and
/// left executable throughout) and its old protection restored, then the instruction cache
/// flushed for it.
///
/// # Safety
/// `at` is mapped code of this process; the caller holds [`SITES`].
unsafe fn patch_code(at: usize, old: u8, new: u8) -> Patch {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut old_protect = 0u32;
    // SAFETY: the page of mapped code holding the byte; its protection is restored below.
    if unsafe { VirtualProtect(at as *const _, 1, PAGE_EXECUTE_READWRITE, &mut old_protect) } == 0 {
        return Patch::Refused;
    }
    // SAFETY: the byte is writable now; only other handlers read it meanwhile.
    let done = unsafe { patch_byte(at as *mut u8, old, new) };
    let mut ignored = 0u32;
    // SAFETY: as above.
    let restored = unsafe { VirtualProtect(at as *const _, 1, old_protect, &mut ignored) } != 0;
    if !restored {
        // the page stays writable; nothing else changes code protection after loading
        tracing::warn!(at = format_args!("{at:#x}"), "a patched code page's protection could not be restored");
    }
    // SAFETY: the instruction cache flush takes any address range of this process.
    unsafe { FlushInstructionCache(GetCurrentProcess(), at as *const _, 1) };
    done
}

/// Patches the console instruction `code` at `at` so that it never traps again
/// ([`replacement`] says with what), once per address.
///
/// Another thread may be running these bytes meanwhile or be about to, and either way it does
/// the right thing. The patch is one byte, and a byte is stored and seen whole by every
/// observer, the instruction fetch included, so a thread meets either the old instruction or
/// the new one, never a mix of the two; and both are whole instructions of the same length,
/// at any alignment (an instruction across an 8-byte word, a cache line or a page is patched
/// the same way), so neither the old code's boundaries nor the new code's differ and no thread
/// can be stopped inside an instruction. With the old byte a thread traps, and its handler
/// then reads the old bytes (emulated as before, and the patch is not tried again) or the new
/// ones (the trap was for bytes that are gone: [`patched_at`] has it run them; the return from
/// the exception serialises the thread, so it fetches them anew). With the new byte it runs a
/// NOP. Old bytes still in a core's prefetch after the store, which Intel's cross-modifying-code
/// rules leave possible until that core serialises, are only another trip through the same
/// handler.
///
/// `code` is a copy of the instruction's bytes, not a slice of the code itself, so what is
/// compared against stays as it was decoded while the code is written. (Other threads'
/// handlers may still read the code plainly during the store: see [`patch_byte`].)
fn patch(at: usize, code: &[u8], what: ConsoleOnly) {
    let Some(rewrite) = replacement(what, code) else { return };
    let mut sites = sites();
    if sites.contains_key(&at) {
        return;
    }
    // SAFETY: `at` is the faulting instruction's address, mapped code, and the byte lies inside
    // the instruction; `SITES` is held.
    let done = unsafe { patch_code(at + rewrite.offset, rewrite.old, rewrite.new) };
    let ok = matches!(done, Patch::Written | Patch::There);
    sites.insert(at, ok.then(|| rewrite.apply(code)));
    if done == Patch::Written {
        PATCHED.fetch_add(1, Relaxed);
    }
    if !ok {
        tracing::debug!(at = format_args!("{at:#x}"), outcome = ?done, instruction = ?what, "a console CPU instruction could not be patched in place: it stays emulated");
    }
}

/// Whether `code` (the bytes at `at` now) is what a patch left there: an illegal-instruction
/// trap for it was raised by the old bytes, before the patch, and the thread runs it again.
/// Bytes of a later different instruction at the same address fail the comparison, so a real
/// fault there is not hidden (the NOPs a patch writes never trap themselves).
fn patched_at(at: usize, code: &[u8]) -> bool {
    matches!(sites().get(&at), Some(Some((new, len))) if code.get(..*len) == Some(&new[..*len]))
}


/// An illegal instruction at `ctx.Rip`: a console-only one is emulated and stepped over
/// (`CONTINUE_EXECUTION`); anything else returns 0 (the crash report and the title's handlers
/// see it).
///
/// # Safety
/// `ctx` is the faulting thread's context, from the exception being dispatched.
pub(crate) unsafe fn emulate(ctx: &mut CONTEXT) -> i32 {
    let rip = ctx.Rip as usize;
    // The same few instructions fault again and again: their bytes are asked about once.
    let known = if remembers() { CODE.with(Cell::get) } else { [(0, 0); 8] };
    let n = match known.iter().find(|&&(at, n)| at == rip && n > 0) {
        Some(&(_, n)) => n,
        None => {
            if !readable(rip, 1) {
                return 0;
            }
            let n = (1..=16).rev().find(|&n| readable(rip, n)).unwrap_or(0);
            if remembers() {
                let mut known = known;
                known.rotate_right(1);
                known[0] = (rip, n);
                CODE.with(|c| c.set(known));
            }
            n
        }
    };
    // SAFETY: `n` bytes at `rip` are readable.
    let code = unsafe { std::slice::from_raw_parts(rip as *const u8, n) };
    let Some((what, len)) = decode::decode(code) else {
        // patched by another thread after this one trapped on the old bytes: run the new ones
        return if patched_at(rip, code) { CONTINUE_EXECUTION } else { 0 };
    };
    static SEEN: std::sync::Mutex<Vec<std::mem::Discriminant<ConsoleOnly>>> = std::sync::Mutex::new(Vec::new());
    if let Ok(mut seen) = SEEN.lock()
        && !seen.contains(&std::mem::discriminant(&what))
    {
        seen.push(std::mem::discriminant(&what));
        if what != ConsoleOnly::Ud2 {
            tracing::warn!(instruction = ?what, at = format_args!("{rip:#x}"), "a console CPU instruction this CPU lacks: emulated (logged once)");
        }
    }
    // the instruction's bytes as they were decoded, for a patch (which writes over the code)
    let mut own = [0u8; 16];
    own[..len].copy_from_slice(&code[..len]);
    match what {
        ConsoleOnly::Ud2 => return 0,
        // nothing to write back on the PC: patched to a hint NOP, no exception next time
        ConsoleOnly::Clwb => patch(rip, &own[..len], what),
        // patched to a hint NOP, then this one is emulated (see `replacement`)
        ConsoleOnly::Monitorx | ConsoleOnly::Mwaitx => {
            if patches_waits() {
                patch(rip, &own[..len], what);
            }
            if what == ConsoleOnly::Mwaitx {
                std::hint::spin_loop();
            }
        }
        ConsoleOnly::Clzero => {
            // with an address-size prefix (0x67) the line is at EAX, zero-extended
            let a32 = code[..len - 3].contains(&0x67);
            let line = (if a32 { ctx.Rax & 0xFFFF_FFFF } else { ctx.Rax } as usize) & !63;
            if !writable(line, 64) {
                return 0; // the real instruction faults there too
            }
            // SAFETY: the 64-byte line is committed and writable.
            unsafe { std::ptr::write_bytes(line as *mut u8, 0, 64) };
        }
        ConsoleOnly::ClzeroLoop { end_reg } => {
            // the whole `clzero; add rax,40h; cmp rax,end; jb` loop at once
            let regs: [u64; 16] = [
                ctx.Rax, ctx.Rcx, ctx.Rdx, ctx.Rbx, ctx.Rsp, ctx.Rbp, ctx.Rsi, ctx.Rdi, ctx.R8, ctx.R9, ctx.R10, ctx.R11, ctx.R12, ctx.R13, ctx.R14, ctx.R15,
            ];
            let end = regs[end_reg as usize & 15] as usize;
            let start = ctx.Rax as usize;
            if end > start && end - start < 2 << 30 {
                // Region by region: a block may lie across several (the title's pages are
                // mapped one view each), and one answer for all of it was a refusal, after
                // which every 64-byte line took an exception of its own.
                let mut at = start;
                while at < end {
                    let Some(to) = writable_to(at, end) else { break };
                    // SAFETY: committed and writable from `at` to `to`.
                    unsafe { std::ptr::write_bytes(at as *mut u8, 0, to - at) };
                    at = to;
                }
                if at == start {
                    return 0;
                }
                ZEROED.fetch_add((at - start) as u64, Relaxed);
                if at < end {
                    // What follows is not writable: the loop goes on there (the same
                    // instruction again), where the real one faults too.
                    ctx.Rax = at as u64;
                    return CONTINUE_EXECUTION;
                }
            }
            ctx.Rax = end as u64;
        }
        ConsoleOnly::Rdpru => {
            let mut v = 0i64;
            // SAFETY: a valid out pointer.
            unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut v) };
            let v = if ctx.Rcx as u32 <= 1 { v as u64 } else { 0 };
            ctx.Rax = v & 0xFFFF_FFFF;
            ctx.Rdx = v >> 32;
        }
        ConsoleOnly::Rdpid(r) => {
            // SAFETY: no preconditions.
            let cpu = unsafe { windows_sys::Win32::System::Threading::GetCurrentProcessorNumber() } as u64;
            let regs: [&mut u64; 16] = [
                &mut ctx.Rax, &mut ctx.Rcx, &mut ctx.Rdx, &mut ctx.Rbx, &mut ctx.Rsp, &mut ctx.Rbp, &mut ctx.Rsi, &mut ctx.Rdi, &mut ctx.R8, &mut ctx.R9, &mut ctx.R10, &mut ctx.R11, &mut ctx.R12, &mut ctx.R13,
                &mut ctx.R14, &mut ctx.R15,
            ];
            *regs[r as usize & 15] = cpu;
        }
        // the low quadword changes; the high one is zeroed, as a Zen 5 does
        ConsoleOnly::Extrq { dst, src, len, idx } => {
            // SAFETY: the context's XMM registers (CONTEXT_FLOATING_POINT is captured for exceptions).
            let x = unsafe { &mut ctx.Anonymous.FltSave.XmmRegisters };
            let (len, idx) = match src {
                Some(s) => {
                    let c = x[s as usize & 15].Low;
                    (c as u8, (c >> 8) as u8)
                }
                None => (len, idx),
            };
            let d = &mut x[dst as usize & 15];
            d.Low = decode::extrq(d.Low, len, idx);
            d.High = 0;
        }
        ConsoleOnly::Insertq { dst, src, reg, len, idx } => {
            // SAFETY: as above.
            let x = unsafe { &mut ctx.Anonymous.FltSave.XmmRegisters };
            let s = x[src as usize & 15];
            let (len, idx) = if reg { (s.High as u8, (s.High >> 8) as u8) } else { (len, idx) };
            let d = &mut x[dst as usize & 15];
            d.Low = decode::insertq(d.Low, s.Low, len, idx);
            d.High = 0;
        }
    }
    ctx.Rip += len as u64;
    KINDS[kind(&what)].fetch_add(1, Relaxed);
    // How often this runs at all (each one is an exception: a trip into Windows and back).
    let emulated = EMULATED.fetch_add(1, Relaxed) + 1;
    if emulated >= 1 << 16 && emulated.is_power_of_two() {
        tracing::info!(
            emulated,
            asked = ASKED.load(Relaxed),
            zeroed_mb = ZEROED.load(Relaxed) >> 20,
            "console CPU instructions emulated so far, the times Windows was asked about memory for them, and what clzero loops zeroed"
        );
    }
    CONTINUE_EXECUTION
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualProtect, VirtualQuery,
    };

    const MWAITX: [u8; 3] = [0x0F, 0x01, 0xFB];
    const MONITORX: [u8; 3] = [0x0F, 0x01, 0xFA];
    /// `clwb [rsp+disp32]`: a SIB and a disp32, nine bytes.
    const CLWB_LONG: [u8; 9] = [0x66, 0x0F, 0xAE, 0xB4, 0x24, 1, 2, 3, 4];

    /// The instruction as [`replacement`] would leave it.
    fn replaced(what: ConsoleOnly, code: &[u8]) -> Option<Vec<u8>> {
        replacement(what, code).map(|r| {
            let (new, len) = r.apply(code);
            new[..len].to_vec()
        })
    }

    #[test]
    fn replacements_by_form() {
        let r = replaced;
        assert_eq!(r(ConsoleOnly::Mwaitx, &MWAITX), Some(vec![0x0F, 0x1F, 0xFB]));
        assert_eq!(r(ConsoleOnly::Monitorx, &MONITORX), Some(vec![0x0F, 0x1F, 0xFA]));
        assert_eq!(replacement(ConsoleOnly::Mwaitx, &MWAITX), Some(Rewrite { offset: 1, old: 0x01, new: 0x1F }));
        // address size, a segment and a REX keep the opcode what it is
        assert_eq!(r(ConsoleOnly::Mwaitx, &[0x67, 0x0F, 0x01, 0xFB]), Some(vec![0x67, 0x0F, 0x1F, 0xFB]));
        assert_eq!(r(ConsoleOnly::Monitorx, &[0x64, 0x67, 0x48, 0x0F, 0x01, 0xFA]), Some(vec![0x64, 0x67, 0x48, 0x0F, 0x1F, 0xFA]));
        // no length limit any more
        let long = [0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x2E, 0x0F, 0x01, 0xFB];
        assert_eq!(replacement(ConsoleOnly::Mwaitx, &long), Some(Rewrite { offset: 7, old: 0x01, new: 0x1F }));
        // 66 / F2 / F3 in front may make it another instruction: emulated
        for p in [0x66, 0xF2, 0xF3] {
            assert_eq!(r(ConsoleOnly::Mwaitx, &[p, 0x0F, 0x01, 0xFB]), None, "{p:02x}");
            assert_eq!(r(ConsoleOnly::Monitorx, &[p, 0x0F, 0x01, 0xFA]), None, "{p:02x}");
            assert_eq!(r(ConsoleOnly::Mwaitx, &[0x48, p, 0x0F, 0x01, 0xFB]), None, "{p:02x} after a REX");
        }
        // clwb [rax], clwb [rsp+8], clwb with a REX, clwb [rsp+disp32]: AE becomes 1F
        assert_eq!(r(ConsoleOnly::Clwb, &[0x66, 0x0F, 0xAE, 0x30]), Some(vec![0x66, 0x0F, 0x1F, 0x30]));
        assert_eq!(r(ConsoleOnly::Clwb, &[0x66, 0x0F, 0xAE, 0x74, 0x24, 0x08]), Some(vec![0x66, 0x0F, 0x1F, 0x74, 0x24, 0x08]));
        assert_eq!(r(ConsoleOnly::Clwb, &[0x66, 0x41, 0x0F, 0xAE, 0x30]), Some(vec![0x66, 0x41, 0x0F, 0x1F, 0x30]));
        assert_eq!(r(ConsoleOnly::Clwb, &CLWB_LONG), Some(vec![0x66, 0x0F, 0x1F, 0xB4, 0x24, 1, 2, 3, 4]));
        assert_eq!(replacement(ConsoleOnly::Clwb, &CLWB_LONG), Some(Rewrite { offset: 2, old: 0xAE, new: 0x1F }));
        assert_eq!(r(ConsoleOnly::Clwb, &[0x64, 0x66, 0x0F, 0xAE, 0x30]), Some(vec![0x64, 0x66, 0x0F, 0x1F, 0x30]));
        // F2 / F3 in front of it, or no 66: emulated
        assert_eq!(r(ConsoleOnly::Clwb, &[0xF3, 0x66, 0x0F, 0xAE, 0x30]), None);
        assert_eq!(r(ConsoleOnly::Clwb, &[0x66, 0xF2, 0x0F, 0xAE, 0x30]), None);
        assert_eq!(r(ConsoleOnly::Clwb, &[0x0F, 0xAE, 0x30]), None);
        assert_eq!(r(ConsoleOnly::Clzero, &[0x0F, 0x01, 0xFC]), None, "clzero is never patched");
        // what the decoder took for them and nothing else is rewritten
        assert_eq!(r(ConsoleOnly::Mwaitx, &[0x0F, 0x1F, 0xFB]), None);
        // and what a patch leaves is no console instruction
        for (what, code) in [(ConsoleOnly::Mwaitx, &MWAITX[..]), (ConsoleOnly::Monitorx, &MONITORX[..]), (ConsoleOnly::Clwb, &CLWB_LONG[..])] {
            assert_eq!(decode::decode(code), Some((what, code.len())));
            assert_eq!(decode::decode(&r(what, code).unwrap()), None);
        }
    }

    /// A buffer of 0xCC with the instruction `code` at `offset` from an 8-byte boundary.
    #[repr(align(8))]
    struct Words([u8; 32]);

    #[test]
    fn a_byte_patch_works_at_every_alignment() {
        for (what, code) in [(ConsoleOnly::Monitorx, &MONITORX[..]), (ConsoleOnly::Mwaitx, &[0x48, 0x0F, 0x01, 0xFB][..]), (ConsoleOnly::Clwb, &CLWB_LONG[..])] {
            // offsets 6 and 7 put the instruction across two words (the title's monitorx at
            // RVA 0x4fb0bd7 is at offset 7 of its word)
            for offset in 0..16 {
                let mut w = Words([0xCC; 32]);
                w.0[offset..offset + code.len()].copy_from_slice(code);
                let rewrite = replacement(what, code).unwrap();
                let at = w.0.as_mut_ptr().wrapping_add(offset + rewrite.offset);
                // SAFETY: our buffer, with no reference into it during the call.
                assert_eq!(unsafe { patch_byte(at, rewrite.old, rewrite.new) }, Patch::Written, "{what:?} at {offset}");
                let mut want = [0xCC; 32];
                let (new, len) = rewrite.apply(code);
                want[offset..offset + len].copy_from_slice(&new[..len]);
                assert_eq!(w.0, want, "{what:?} at {offset}");
            }
        }
    }

    #[test]
    fn a_byte_patch_leaves_other_bytes_alone_and_is_written_once() {
        let mut w = Words([0xCC; 32]);
        w.0[7..10].copy_from_slice(&MWAITX);
        let base = w.0.as_mut_ptr();
        // SAFETY: our buffer, with no reference into it during the calls.
        unsafe {
            // the byte there is not the one decoded (rewritten meanwhile): left alone
            assert_eq!(patch_byte(base.add(7), 0x01, 0x1F), Patch::Changed);
            assert_eq!(w.0[7], 0x0F);
            assert_eq!(patch_byte(base.add(8), 0x01, 0x1F), Patch::Written);
            assert_eq!(patch_byte(base.add(8), 0x01, 0x1F), Patch::There, "a second thread's try");
        }
        let mut want = [0xCC; 32];
        want[7..10].copy_from_slice(&[0x0F, 0x1F, 0xFB]);
        assert_eq!(w.0, want);
    }

    /// A page of code, read and execute only as the title's is, with pieces of code at offsets.
    struct CodePage(*mut u8);

    // SAFETY: the page is only read and run by other threads, and written through patch_code.
    unsafe impl Send for CodePage {}
    // SAFETY: as above.
    unsafe impl Sync for CodePage {}

    impl CodePage {
        fn new(pieces: &[(usize, &[u8])]) -> Self {
            // SAFETY: a fresh page, written while it is ours and writable, then made code.
            unsafe {
                let page = VirtualAlloc(std::ptr::null(), 0x1000, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE).cast::<u8>();
                assert!(!page.is_null());
                page.write_bytes(0xCC, 0x1000);
                for &(offset, code) in pieces {
                    std::ptr::copy_nonoverlapping(code.as_ptr(), page.add(offset), code.len());
                }
                let mut old = 0;
                assert_ne!(VirtualProtect(page.cast(), 0x1000, PAGE_EXECUTE_READ, &mut old), 0);
                Self(page)
            }
        }

        fn at(&self, offset: usize) -> usize {
            self.0 as usize + offset
        }

        fn bytes(&self, offset: usize, n: usize) -> Vec<u8> {
            // SAFETY: inside our page, which is readable.
            unsafe { std::slice::from_raw_parts(self.0.add(offset), n).to_vec() }
        }

        fn protection(&self) -> u32 {
            // SAFETY: plain data filled in by the query, for our page.
            unsafe {
                let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                assert_ne!(VirtualQuery(self.0.cast(), &mut m, size_of_val(&m)), 0);
                m.Protect
            }
        }

        /// Runs the code at `offset` as a function (it ends in `ret`).
        fn call(&self, offset: usize) {
            // SAFETY: the bytes there are instructions that end in `ret` and touch nothing.
            let f: extern "C" fn() = unsafe { std::mem::transmute(self.0.add(offset)) };
            f();
        }
    }

    impl Drop for CodePage {
        fn drop(&mut self) {
            // SAFETY: our page, no longer used.
            unsafe { VirtualFree(self.0.cast(), 0, MEM_RELEASE) };
        }
    }

    #[test]
    fn code_is_patched_and_its_protection_restored() {
        let mut clwb = CLWB_LONG.to_vec();
        clwb.push(0xC3);
        // mwaitx at offset 7 of its word, clwb [rax] at offset 6, the long clwb across a word
        let page = CodePage::new(&[(0x17, &[0x0F, 0x01, 0xFB, 0xC3]), (0x26, &[0x66, 0x0F, 0xAE, 0x30, 0xC3]), (0x3C, &clwb)]);
        let held = sites();
        // SAFETY: our code page; `SITES` is held.
        unsafe {
            assert_eq!(patch_code(page.at(0x18), 0x01, 0x1F), Patch::Written);
            assert_eq!(patch_code(page.at(0x28), 0xAE, 0x1F), Patch::Written);
            assert_eq!(patch_code(page.at(0x3E), 0xAE, 0x1F), Patch::Written);
            assert_eq!(patch_code(page.at(0x18), 0x01, 0x1F), Patch::There);
        }
        drop(held);
        assert_eq!(page.bytes(0x17, 4), [0x0F, 0x1F, 0xFB, 0xC3]);
        assert_eq!(page.bytes(0x26, 5), [0x66, 0x0F, 0x1F, 0x30, 0xC3]);
        assert_eq!(page.bytes(0x3C, 10), [0x66, 0x0F, 0x1F, 0xB4, 0x24, 1, 2, 3, 4, 0xC3]);
        assert_eq!(page.protection(), PAGE_EXECUTE_READ);
        // nop ebx, ret: runs on any CPU now
        page.call(0x17);
        // nop word [rax] and nop word [rsp+0x04030201] with whatever is in the registers: the
        // NOPs read no memory, and the ret right after them is reached, so the length held
        page.call(0x26);
        page.call(0x3C);
    }

    /// The handler as a trapping thread meets it: the first trap patches and is emulated, a
    /// trap raised by the old bytes but handled after the patch runs the new ones, an
    /// instruction across two words is patched like any other, and a NOP that no patch wrote
    /// is not taken for one.
    #[test]
    fn the_handler_patches_once_and_reruns_patched_bytes() {
        assert!(patches_waits(), "the tests run with VICTORY_PATCH_WAITS unset");
        let page = CodePage::new(&[(0x40, &[0x48, 0x0F, 0x01, 0xFA, 0x0F, 0x01, 0xFB, 0xC3]), (0x5E, &[0x0F, 0x01, 0xFB, 0xC3]), (0x67, &[0x66, 0x0F, 0xAE, 0x30, 0xC3])]);
        // the very bytes a patch leaves, at an address no patch was at
        let other = CodePage::new(&[(0, &[0x0F, 0x1F, 0xFB]), (8, &[0x90, 0x90, 0x90])]);
        // SAFETY: plain data; the handler reads only our pages through Rip.
        let mut ctx: CONTEXT = unsafe { std::mem::zeroed() };
        let mut trap = |at: usize| {
            ctx.Rip = at as u64;
            // SAFETY: a context of our own whose Rip points into one of our readable pages.
            let r = unsafe { emulate(&mut ctx) };
            (r, ctx.Rip as usize - at)
        };
        assert_eq!(trap(page.at(0x40)), (CONTINUE_EXECUTION, 4), "REX monitorx emulated");
        assert_eq!(trap(page.at(0x44)), (CONTINUE_EXECUTION, 3), "mwaitx emulated");
        assert_eq!(page.bytes(0x40, 8), [0x48, 0x0F, 0x1F, 0xFA, 0x0F, 0x1F, 0xFB, 0xC3]);
        // traps raised before the patch, handled after it: run again where they were
        assert_eq!(trap(page.at(0x40)), (CONTINUE_EXECUTION, 0));
        assert_eq!(trap(page.at(0x44)), (CONTINUE_EXECUTION, 0));
        // offset 6 of its word: emulated and patched, then run again
        assert_eq!(trap(page.at(0x5E)), (CONTINUE_EXECUTION, 3));
        assert_eq!(page.bytes(0x5E, 3), [0x0F, 0x1F, 0xFB]);
        assert_eq!(trap(page.at(0x5E)), (CONTINUE_EXECUTION, 0));
        // clwb [rax] at offset 7: stepped over and patched
        assert_eq!(trap(page.at(0x67)), (CONTINUE_EXECUTION, 4));
        assert_eq!(page.bytes(0x67, 4), [0x66, 0x0F, 0x1F, 0x30]);
        assert_eq!(trap(page.at(0x67)), (CONTINUE_EXECUTION, 0));
        // NOPs that no patch wrote are no console instruction: the fault is not hidden
        assert_eq!(trap(other.at(0)), (0, 0));
        assert_eq!(trap(other.at(8)), (0, 0));
        page.call(0x40);
        page.call(0x5E);
        page.call(0x67);
    }

    /// Other threads run the bytes while one of them is rewritten again and again and the
    /// page's protection changes under them, so a thread that saw its page without execute
    /// rights, or a byte half written, would fault here. The byte flipped is the ModRM of a
    /// hint NOP across two words (`nop edx` and `nop ebx`), so both states run on any CPU.
    #[test]
    fn code_is_patched_while_other_threads_run_it() {
        let page = std::sync::Arc::new(CodePage::new(&[(0x87, &[0x0F, 0x1F, 0xFA, 0xC3])]));
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let runners: Vec<_> = (0..3)
            .map(|_| {
                let (page, stop) = (page.clone(), stop.clone());
                std::thread::spawn(move || {
                    while !stop.load(Relaxed) {
                        page.call(0x87);
                    }
                })
            })
            .collect();
        for i in 0..500 {
            let (old, new) = if i % 2 == 0 { (0xFA, 0xFB) } else { (0xFB, 0xFA) };
            let _held = sites();
            // SAFETY: our code page; `SITES` is held.
            assert_eq!(unsafe { patch_code(page.at(0x89), old, new) }, Patch::Written);
        }
        stop.store(true, Relaxed);
        for r in runners {
            r.join().unwrap();
        }
        assert_eq!(page.bytes(0x87, 4), [0x0F, 0x1F, 0xFA, 0xC3]);
        assert_eq!(page.protection(), PAGE_EXECUTE_READ);
    }
}
