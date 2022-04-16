// This file is part of libfringe, a low-level green threading library.
// Copyright (c) edef <edef@edef.eu>,
//               whitequark <whitequark@whitequark.org>
//               Amanieu d'Antras <amanieu@gmail.com>
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// To understand the code in this file, keep in mind these two facts:
// * x86_64 SysV C ABI has a "red zone": 128 bytes under the top of the stack
//   that is defined to be unmolested by signal handlers, interrupts, etc.
//   Leaf functions can use the red zone without adjusting rsp or rbp.
// * x86_64 SysV C ABI requires the stack to be aligned at function entry,
//   so that (%rsp+8) is a multiple of 16. Aligned operands are a requirement
//   of SIMD instructions, and making this the responsibility of the caller
//   avoids having to maintain a frame pointer, which is necessary when
//   a function has to realign the stack from an unknown state.
// * x86_64 SysV C ABI passes the first argument in %rdi. We also use %rdi
//   to pass a value while swapping context; this is an arbitrary choice
//   (we clobber all registers and could use any of them) but this allows us
//   to reuse the swap function to perform the initial call. We do the same
//   thing with %rsi to pass the stack pointer to the new context.
//
// To understand the DWARF CFI code in this file, keep in mind these facts:
// * CFI is "call frame information"; a set of instructions to a debugger or
//   an unwinder that allow it to simulate returning from functions. This implies
//   restoring every register to its pre-call state, as well as the stack pointer.
// * CFA is "call frame address"; the value of stack pointer right before the call
//   instruction in the caller. Everything strictly below CFA (and inclusive until
//   the next CFA) is the call frame of the callee. This implies that the return
//   address is the part of callee's call frame.
// * Logically, DWARF CFI is a table where rows are instruction pointer values and
//   columns describe where registers are spilled (mostly using expressions that
//   compute a memory location as CFA+n). A .cfi_offset pseudoinstruction changes
//   the state of a column for all IP numerically larger than the one it's placed
//   after. A .cfi_def_* pseudoinstruction changes the CFA value similarly.
// * Simulating return is as easy as restoring register values from the CFI table
//   and then setting stack pointer to CFA.
//
// A high-level overview of the function of the trampolines when unwinding is:
// * The 2nd init trampoline puts a controlled value (written in swap to `new_cfa`)
//   into %rbp. This is then used as the CFA for the 1st trampoline.
// * This controlled value points to the bottom of the stack of the parent context,
//   which holds the saved %rbp and return address from the call to swap().
// * The 1st init trampoline tells the unwinder to restore %rbp and its return
//   address from the stack frame at %rbp (in the parent stack), thus continuing
//   unwinding at the swap call site instead of falling off the end of context stack.

use crate::{arch::StackPointer, unwind};
use core::{arch::asm, ptr::NonNull};

pub const STACK_ALIGNMENT: usize = 16;

pub unsafe fn init(
  stack_base: *mut u8,
  f: unsafe extern "C" fn(usize, StackPointer),
) -> StackPointer {
  #[cfg(not(target_vendor = "apple"))]
  #[allow(named_asm_labels)]
  #[naked]
  unsafe extern "C" fn trampoline_1() {
    asm!(
      // gdb has a hardcoded check that rejects backtraces where frame addresses
      // do not monotonically decrease. It is turned off if the function is called
      // "__morestack" and that is hardcoded. So, to make gdb backtraces match
      // the actual unwinder behavior, we call ourselves "__morestack" and mark
      // the symbol as local; it shouldn't interfere with anything.
      "__morestack:",
      ".local __morestack",
      // Set up the first part of our DWARF CFI linking stacks together. When
      // we reach this function from unwinding, %rbp will be pointing at the bottom
      // of the parent linked stack. This link is set each time swap() is called.
      // When unwinding the frame corresponding to this function, a DWARF unwinder
      // will use %rbp+16 as the next call frame address, restore return address
      // from CFA-8 and restore %rbp from CFA-16. This mirrors what the second half
      // of `swap_trampoline` does.
      ".cfi_def_cfa rbp, 16",
      ".cfi_offset rbp, -16",
      // This nop is here so that the initial swap doesn't return to the start
      // of the trampoline, which confuses the unwinder since it will look for
      // frame information in the previous symbol rather than this one. It is
      // never actually executed.
      "nop",
      // Stack unwinding in some versions of libunwind doesn't seem to like
      // 1-byte symbols, so we add a second nop here. This instruction isn't
      // executed either, it is only here to pad the symbol size.
      "nop",
      ".Lend:",
      ".size __morestack, .Lend-__morestack",
      options(noreturn)
    );
  }

  #[cfg(target_vendor = "apple")]
  #[naked]
  unsafe extern "C" fn trampoline_1() {
    asm!(
      // Identical to the above, except avoids .local/.size that aren't available on Mach-O.
      "__morestack:",
      ".private_extern __morestack",
      ".cfi_def_cfa rbp, 16",
      ".cfi_offset rbp, -16",
      "nop",
      "nop",
      options(noreturn),
    )
  }

  #[naked]
  unsafe extern "C" fn trampoline_2() {
    asm!(
      // Set up the second part of our DWARF CFI.
      // When unwinding the frame corresponding to this function, a DWARF unwinder
      // will restore %rbp (and thus CFA of the first trampoline) from the stack slot.
      // This stack slot is updated every time swap() is called to point to the bottom
      // of the stack of the context switch just switched from.
      ".cfi_def_cfa rbp, 16",
      ".cfi_offset rbp, -16",

      // This nop is here so that the return address of the swap trampoline
      // doesn't point to the start of the symbol. This confuses gdb's backtraces,
      // causing them to think the parent function is trampoline_1 instead of
      // trampoline_2.
      "nop",

      // Call unwind_wrapper with the provided function and the stack base address.
      "lea    rdx, [rsp + 32]",
      "mov    rcx, [rsp + 16]",
      "call   {0}",

      // Restore the stack pointer of the parent context. No CFI adjustments
      // are needed since we have the same stack frame as trampoline_1.
      "mov    rsp, [rsp]",

      // Restore frame pointer of the parent context.
      "pop    rbp",
      ".cfi_adjust_cfa_offset -8",
      ".cfi_restore rbp",

      // If the returned value is nonzero, trigger an unwind in the parent
      // context with the given exception object.
      "mov    rdi, rax",
      "test   rax, rax",
      "jnz    {1}",

      // Clear the stack pointer. We can't call into this context any more once
      // the function has returned.
      "xor    rsi, rsi",

      // Return into the parent context. Use `pop` and `jmp` instead of a `ret`
      // to avoid return address mispredictions (~8ns per `ret` on Ivy Bridge).
      "pop    rax",
      ".cfi_adjust_cfa_offset -8",
      ".cfi_register rip, rax",
      "jmp    rax",

      sym unwind::unwind_wrapper,
      sym unwind::start_unwind,

      options(noreturn),
    );
  }

  // We set up the stack in a somewhat special way so that to the unwinder it
  // looks like trampoline_1 has called trampoline_2, which has in turn called
  // swap::trampoline.
  //
  // There are 2 call frames in this setup, each containing the return address
  // followed by the %rbp value for that frame. This setup supports unwinding
  // using DWARF CFI as well as the frame pointer-based unwinding used by tools
  // such as perf or dtrace.
  let mut sp = StackPointer::new(stack_base);

  sp.push(0 as usize); // Padding to ensure the stack is properly aligned
  sp.push(f as usize); // Function that trampoline_2 should call

  // Call frame for trampoline_2. The CFA slot is updated by swap::trampoline
  // each time a context switch is performed.
  sp.push(trampoline_1 as usize + 2); // Return after the 2 nops
  sp.push(0xdeaddeaddead0cfa); // CFA slot

  // Call frame for swap::trampoline. We set up the %rbp value to point to the
  // parent call frame.
  let frame = sp.offset(0);
  sp.push(trampoline_2 as usize + 1); // Entry point, skip initial nop
  sp.push(frame as usize); // Pointer to parent call frame

  sp
}

#[inline(always)]
pub unsafe fn swap_link(
  arg: usize,
  new_sp: StackPointer,
  new_stack_base: *mut u8,
) -> (usize, Option<StackPointer>) {
  let mut ret: usize;
  let mut ret_sp: *mut usize;

  asm!(
      // FIXME(cynecx): figure out correct cfi directives.
      // Save `rbx` because we can't use `rbx` in a clobber because it's reserved by llvm.
      // "push rbx",
      // ".cfi_adjust_cfa_offset 8",
      // ".cfi_rel_offset rbx, 0",
      // Push the return address
      "lea    rax, [rip + 0f]",
      "push   rax",
      // Save frame pointer explicitly; the unwinder uses it to find CFA of
      // the caller, and so it has to have the correct value immediately after
      // the call instruction that invoked the trampoline.
      "push   rbp",
      // Link the call stacks together by writing the current stack bottom
      // address to the CFA slot in the new stack.
      "mov    [rcx - 32], rsp",
      // Pass the stack pointer of the old context to the new one.
      "mov    rsi, rsp",
      // Load stack pointer of the new context.
      "mov    rsp, rdx",
      // Restore frame pointer of the new context.
      "pop    rbp",
      // Return into the new context. Use `pop` and `jmp` instead of a `ret`
      // to avoid return address mispredictions (~8ns per `ret` on Ivy Bridge).
      "pop    rax",
      "jmp    rax",
      // Reentry
      "0:",
      // FIXME(cynecx): figure out correct cfi directives.
      // Restore `rbx` which we've saved before because we can't use it as a clobber.
      // "pop    rbx",
      // ".cfi_adjust_cfa_offset -8",
      // ".cfi_restore rbx",
      // Outputs
      lateout("rdi") ret,
      lateout("rsi") ret_sp,
      // Inputs
      in("rdi") arg,
      in("rdx") new_sp.offset(0),
      in("rcx") new_stack_base,
      // Clobbers
      out("rax") _, out("rbx") _, lateout("rcx") _, lateout("rdx") _,
      out("r8") _, out("r9") _, out("r10") _, out("r11") _,
      out("r12") _, out("r13") _, out("r14") _, out("r15") _,
      out("mm0") _, out("mm1") _, out("mm2") _, out("mm3") _,
      out("mm4") _, out("mm5") _, out("mm6") _, out("mm7") _,
      out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
      out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
      out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
      out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
      out("xmm16") _, out("xmm17") _, out("xmm18") _, out("xmm19") _,
      out("xmm20") _, out("xmm21") _, out("xmm22") _, out("xmm23") _,
      out("xmm24") _, out("xmm25") _, out("xmm26") _, out("xmm27") _,
      out("xmm28") _, out("xmm29") _, out("xmm30") _, out("xmm31") _,
      /* Options:
          rustc emits the following clobbers,
          - by *not* specifying `options(preserves_flags)`:
              (x86) ~{dirflag},~{flags},~{fpsr}
              (ARM/AArch64) ~{cc}
          - by *not* specifying `options(nomem)`:
              ~{memory}
          - by *not* specifying `nostack`:
              alignstack
      */
      options(unwind)
  );
  (ret, NonNull::new(ret_sp).map(StackPointer::from))
}

#[inline(always)]
pub unsafe fn swap(arg: usize, new_sp: StackPointer) -> (usize, StackPointer) {
  // This is identical to swap_link, but without the write to the CFA slot.
  let mut ret: usize;
  let mut ret_sp: *mut usize;

  asm!(
      // FIXME(cynecx): figure out correct cfi directives.
      // Save `rbx` because we can't use `rbx` in a clobber because it's reserved by llvm.
      // "push rbx",
      // ".cfi_adjust_cfa_offset 8",
      // ".cfi_rel_offset rbx, 0",
      // Push the return address
      "lea    rax, [rip + 0f]",
      "push   rax",
      // Save frame pointer explicitly; the unwinder uses it to find CFA of
      // the caller, and so it has to have the correct value immediately after
      // the call instruction that invoked the trampoline.
      "push   rbp",
      // Pass the stack pointer of the old context to the new one.
      "mov    rsi, rsp",
      // Load stack pointer of the new context.
      "mov    rsp, rdx",
      // Restore frame pointer of the new context.
      "pop    rbp",
      // Return into the new context. Use `pop` and `jmp` instead of a `ret`
      // to avoid return address mispredictions (~8ns per `ret` on Ivy Bridge).
      "pop    rax",
      "jmp    rax",
      // Reentry
      "0:",
      // FIXME(cynecx): figure out correct cfi directives.
      // Restore `rbx` which we've saved before because we can't use it as a clobber.
      // "pop    rbx",
      // ".cfi_adjust_cfa_offset -8",
      // ".cfi_restore rbx",
      //
      // Outputs
      lateout("rdi") ret,
      lateout("rsi") ret_sp,
      // Inputs
      in("rdi") arg,
      in("rdx") new_sp.offset(0),
      // Clobbers
      out("rax") _, out("rbx") _, out("rcx") _, lateout("rdx") _,
      out("r8") _, out("r9") _, out("r10") _, out("r11") _,
      out("r12") _, out("r13") _, out("r14") _, out("r15") _,
      out("mm0") _, out("mm1") _, out("mm2") _, out("mm3") _,
      out("mm4") _, out("mm5") _, out("mm6") _, out("mm7") _,
      out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
      out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
      out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
      out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
      out("xmm16") _, out("xmm17") _, out("xmm18") _, out("xmm19") _,
      out("xmm20") _, out("xmm21") _, out("xmm22") _, out("xmm23") _,
      out("xmm24") _, out("xmm25") _, out("xmm26") _, out("xmm27") _,
      out("xmm28") _, out("xmm29") _, out("xmm30") _, out("xmm31") _,
      /* Options:
          rustc emits the following clobbers,
          - by *not* specifying `options(preserves_flags)`:
              (x86) ~{dirflag},~{flags},~{fpsr}
              (ARM/AArch64) ~{cc}
          - by *not* specifying `options(nomem)`:
              ~{memory}
          - by *not* specifying `nostack`:
              alignstack
      */
      options(unwind)
  );

  (ret, StackPointer::new(ret_sp))
}

#[inline(always)]
pub unsafe fn unwind(new_sp: StackPointer, new_stack_base: *mut u8) {
  // Argument to pass to start_unwind, based on the stack base address.
  let arg = unwind::unwind_arg(new_stack_base);

  // This is identical to swap_link, except that it performs a tail call to
  // start_unwind instead of returning into the target context.
  asm!(
      // FIXME(cynecx): figure out correct cfi directives.
      // Save `rbx` because we can't use `rbx` in a clobber because it's reserved by llvm.
      // "push rbx",
      // ".cfi_adjust_cfa_offset 8",
      // ".cfi_rel_offset rbx, 0",
      // Push the return address
      "lea    rax, [rip + 0f]",
      "push   rax",
      // Save frame pointer explicitly; the unwinder uses it to find CFA of
      // the caller, and so it has to have the correct value immediately after
      // the call instruction that invoked the trampoline.
      "push   rbp",
      // Link the call stacks together by writing the current stack bottom
      // address to the CFA slot in the new stack.
      "mov    [rcx - 32], rsp",
      // Load stack pointer of the new context.
      "mov    rsp, rdx",
      // Restore frame pointer of the new context.
      "pop    rbp",
      // Jump to the start_unwind function, which will force a stack unwind in
      // the target context. This will eventually return to us through the
      // stack link.
      "jmp    {0}",
      // Reentry
      "0:",
      // FIXME(cynecx): figure out correct cfi directives.
      // Restore `rbx` which we've saved before because we can't use it as a clobber.
      // "pop    rbx",
      // ".cfi_adjust_cfa_offset -8",
      // ".cfi_restore rbx",
      // Symbols
      sym unwind::start_unwind,
      // Inputs
      in("rdi") arg,
      in("rdx") new_sp.offset(0),
      in("rcx") new_stack_base,
      // Clobbers
      out("rax") _, out("rbx") _, lateout("rcx") _, lateout("rdx") _,
      lateout("rdi") _, lateout("rsi") _,
      out("r8") _, out("r9") _, out("r10") _, out("r11") _,
      out("r12") _, out("r13") _, out("r14") _, out("r15") _,
      out("mm0") _, out("mm1") _, out("mm2") _, out("mm3") _,
      out("mm4") _, out("mm5") _, out("mm6") _, out("mm7") _,
      out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _,
      out("xmm4") _, out("xmm5") _, out("xmm6") _, out("xmm7") _,
      out("xmm8") _, out("xmm9") _, out("xmm10") _, out("xmm11") _,
      out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
      out("xmm16") _, out("xmm17") _, out("xmm18") _, out("xmm19") _,
      out("xmm20") _, out("xmm21") _, out("xmm22") _, out("xmm23") _,
      out("xmm24") _, out("xmm25") _, out("xmm26") _, out("xmm27") _,
      out("xmm28") _, out("xmm29") _, out("xmm30") _, out("xmm31") _,
      /* Options:
          rustc emits the following clobbers,
          - by *not* specifying `options(preserves_flags)`:
              (x86) ~{dirflag},~{flags},~{fpsr}
              (ARM/AArch64) ~{cc}
          - by *not* specifying `options(nomem)`:
              ~{memory}
          - by *not* specifying `nostack`:
              alignstack
      */
      options(unwind)
  );
}
