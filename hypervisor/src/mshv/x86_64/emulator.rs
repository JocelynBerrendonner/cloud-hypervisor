// SPDX-License-Identifier: Apache-2.0 OR BSD-3-Clause
//
// Copyright © 2024, Microsoft Corporation
//

use anyhow::anyhow;
use iced_x86::Register;
use log::debug;
use log::info;
use log::warn;
use mshv_bindings::*;
use std::cell::Cell;

thread_local! {
    static LAST_MMIO_READ_GPA: Cell<u64> = Cell::new(0);
    static MMIO_READ_COUNT: Cell<u64> = Cell::new(0);
    static LAST_MMIO_READ_DATA: Cell<u64> = Cell::new(0);
}

use crate::arch::emulator::{PlatformEmulator, PlatformError};
use crate::arch::x86::emulator::{CpuStateManager, EmulatorCpuState};
use crate::cpu::Vcpu;
use crate::mshv::MshvVcpu;

pub struct MshvEmulatorContext<'a> {
    pub vcpu: &'a MshvVcpu,
    pub map: (u64, u64), // Initial GVA to GPA mapping provided by the hypervisor
}

impl MshvEmulatorContext<'_> {
    // Do the actual gva -> gpa translation
    #[allow(non_upper_case_globals)]
    fn translate(&self, gva: u64, flags: u32) -> Result<u64, PlatformError> {
        if self.map.0 == gva {
            return Ok(self.map.1);
        }

        let (gpa, result_code) = self
            .vcpu
            .translate_gva(gva, flags.into())
            .map_err(|e| PlatformError::TranslateVirtualAddress(anyhow!(e)))?;

        match result_code {
            hv_translate_gva_result_code_HV_TRANSLATE_GVA_SUCCESS => Ok(gpa),
            _ => Err(PlatformError::TranslateVirtualAddress(anyhow!(result_code))),
        }
    }

    fn r(&self, gva: u64, data: &mut [u8], flags: u32) -> Result<(), PlatformError> {
        let gpa = self.translate(gva, flags)?;
        debug!(
            "mshv emulator: memory read {} bytes from [{:#x} -> {:#x}]",
            data.len(),
            gva,
            gpa
        );

        if let Some(vm_ops) = &self.vcpu.vm_ops
            && vm_ops.guest_mem_read(gpa, data).is_err()
        {
            info!(
                "[MMIO-DIAG] emulator read: gva=0x{:x} gpa=0x{:x} len={} (MMIO path)",
                gva, gpa, data.len(),
            );
            vm_ops
                .mmio_read(gpa, data)
                .map_err(|e| PlatformError::MemoryReadFailure(e.into()))?;
            if data.len() <= 8 {
                info!(
                    "[MMIO-DIAG] emulator read result: gpa=0x{:x} data={:02x?}",
                    gpa, data,
                );
            }

            // Polling detector: track repeated reads to the same GPA
            let data_val = {
                let mut buf = [0u8; 8];
                let n = data.len().min(8);
                buf[..n].copy_from_slice(&data[..n]);
                u64::from_le_bytes(buf)
            };
            LAST_MMIO_READ_GPA.with(|last| {
                MMIO_READ_COUNT.with(|count| {
                    LAST_MMIO_READ_DATA.with(|last_data| {
                        if last.get() == gpa {
                            let c = count.get() + 1;
                            count.set(c);
                            last_data.set(data_val);
                            if c == 10 || c == 100 || c == 1000 || c % 10000 == 0 {
                                warn!(
                                    "[MMIO-DIAG] POLL DETECTED: gpa=0x{:x} page_offset=0x{:x} \
                                     read #{} times, data=0x{:x} all_ff={}",
                                    gpa, gpa & 0xFFF, c, data_val,
                                    data_val == u64::MAX || (data.len() == 4 && data_val as u32 == u32::MAX),
                                );
                            }
                        } else {
                            if count.get() > 5 {
                                warn!(
                                    "[MMIO-DIAG] POLL END: gpa=0x{:x} page_offset=0x{:x} \
                                     was read {} times, last_data=0x{:x}",
                                    last.get(), last.get() & 0xFFF,
                                    count.get(), last_data.get(),
                                );
                            }
                            last.set(gpa);
                            count.set(1);
                            last_data.set(data_val);
                        }
                    });
                });
            });
        }

        Ok(())
    }

    fn read_memory_flags(
        &self,
        gva: u64,
        data: &mut [u8],
        flags: u32,
    ) -> Result<(), PlatformError> {
        let mut len = data.len() as u64;

        // Compare the page number of the first and last byte. If they are different, this is a
        // cross-page access.
        let pg1 = gva >> HV_HYP_PAGE_SHIFT;
        let pg2 = (gva + len - 1) >> HV_HYP_PAGE_SHIFT;
        let cross_page = pg1 != pg2;

        if cross_page {
            // We only handle one page cross-page access
            assert!(pg1 + 1 == pg2);
            let n = (gva + len) & HV_HYP_PAGE_MASK as u64;
            len -= n;
        }

        self.r(gva, &mut data[..len as usize], flags)?;

        if cross_page {
            self.r(gva + len, &mut data[len as usize..], flags)?;
        }

        Ok(())
    }

    fn w(&mut self, gva: u64, data: &[u8]) -> Result<(), PlatformError> {
        let gpa = self.translate(gva, HV_TRANSLATE_GVA_VALIDATE_WRITE)?;
        debug!(
            "mshv emulator: memory write {} bytes at [{:#x} -> {:#x}]",
            data.len(),
            gva,
            gpa
        );

        if let Some(vm_ops) = &self.vcpu.vm_ops
            && vm_ops.guest_mem_write(gpa, data).is_err()
        {
            if data.len() <= 8 {
                info!(
                    "[MMIO-DIAG] emulator write: gva=0x{:x} gpa=0x{:x} len={} data={:02x?} (MMIO path)",
                    gva, gpa, data.len(), data,
                );
            } else {
                info!(
                    "[MMIO-DIAG] emulator write: gva=0x{:x} gpa=0x{:x} len={} (MMIO path)",
                    gva, gpa, data.len(),
                );
            }
            vm_ops
                .mmio_write(gpa, data)
                .map_err(|e| PlatformError::MemoryWriteFailure(e.into()))?;
        }

        Ok(())
    }

    pub fn update_cpu_state(
        &self,
        cpu_id: usize,
        old_state: <Self as PlatformEmulator>::CpuState,
        new_state: <Self as PlatformEmulator>::CpuState,
    ) -> Result<(), PlatformError> {
        if cpu_id != self.vcpu.vp_index as usize {
            return Err(PlatformError::SetCpuStateFailure(anyhow!(
                "CPU id mismatch {:?} {:?}",
                cpu_id,
                self.vcpu.vp_index
            )));
        }

        debug!("mshv emulator: Updating CPU state");
        debug!("mshv emulator: {:#x?}", new_state.regs);

        self.vcpu
            .set_regs(&new_state.regs)
            .map_err(|e| PlatformError::SetCpuStateFailure(e.into()))?;

        if old_state.sregs != new_state.sregs {
            debug!("mshv emulator: Updating CPU special registers");
            debug!("mshv emulator: {:#x?}", new_state.sregs);
            self.vcpu
                .set_sregs(&new_state.sregs)
                .map_err(|e| PlatformError::SetCpuStateFailure(e.into()))?;
        }

        Ok(())
    }
}

/// Platform emulation for Hyper-V
impl PlatformEmulator for MshvEmulatorContext<'_> {
    type CpuState = EmulatorCpuState;

    fn read_memory(&self, gva: u64, data: &mut [u8]) -> Result<(), PlatformError> {
        self.read_memory_flags(gva, data, HV_TRANSLATE_GVA_VALIDATE_READ)
    }

    fn write_memory(&mut self, gva: u64, data: &[u8]) -> Result<(), PlatformError> {
        let mut len = data.len() as u64;

        // Compare the page number of the first and last byte. If they are different, this is a
        // cross-page access.
        let pg1 = gva >> HV_HYP_PAGE_SHIFT;
        let pg2 = (gva + len - 1) >> HV_HYP_PAGE_SHIFT;
        let cross_page = pg1 != pg2;

        if cross_page {
            // We only handle one page cross-page access
            assert!(pg1 + 1 == pg2);
            let n = (gva + len) & HV_HYP_PAGE_MASK as u64;
            len -= n;
        }

        self.w(gva, &data[..len as usize])?;

        if cross_page {
            self.w(gva + len, &data[len as usize..])?;
        }

        Ok(())
    }

    fn cpu_state(&self, cpu_id: usize) -> Result<Self::CpuState, PlatformError> {
        if cpu_id != self.vcpu.vp_index as usize {
            return Err(PlatformError::GetCpuStateFailure(anyhow!(
                "CPU id mismatch {:?} {:?}",
                cpu_id,
                self.vcpu.vp_index
            )));
        }

        let regs = self
            .vcpu
            .get_regs()
            .map_err(|e| PlatformError::GetCpuStateFailure(e.into()))?;
        let sregs = self
            .vcpu
            .get_sregs()
            .map_err(|e| PlatformError::GetCpuStateFailure(e.into()))?;

        debug!("mshv emulator: Getting new CPU state");
        debug!("mshv emulator: {regs:#x?}");

        Ok(EmulatorCpuState { regs, sregs })
    }

    fn set_cpu_state(&self, cpu_id: usize, state: Self::CpuState) -> Result<(), PlatformError> {
        if cpu_id != self.vcpu.vp_index as usize {
            return Err(PlatformError::SetCpuStateFailure(anyhow!(
                "CPU id mismatch {:?} {:?}",
                cpu_id,
                self.vcpu.vp_index
            )));
        }

        debug!("mshv emulator: Setting new CPU state");
        debug!("mshv emulator: {:#x?}", state.regs);

        self.vcpu
            .set_regs(&state.regs)
            .map_err(|e| PlatformError::SetCpuStateFailure(e.into()))?;
        self.vcpu
            .set_sregs(&state.sregs)
            .map_err(|e| PlatformError::SetCpuStateFailure(e.into()))
    }

    fn fetch(&self, ip: u64, instruction_bytes: &mut [u8]) -> Result<(), PlatformError> {
        let rip =
            self.cpu_state(self.vcpu.vp_index as usize)?
                .linearize(Register::CS, ip, false)?;
        self.read_memory_flags(
            rip,
            instruction_bytes,
            HV_TRANSLATE_GVA_VALIDATE_READ | HV_TRANSLATE_GVA_VALIDATE_EXECUTE,
        )
    }
}
