from unicorn import *
from unicorn.arm_const import *
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import unicorn


SEPARATION_SLC = 0
SEPARATION_FLEX = 2
SLC_PAGES_PER_BLOCK = 64
MLC_PAGES_PER_BLOCK = 128

# --- OneNAND address window layout -----------------------------------------
# The original code mapped the whole 0x30000000-0x30020000 window as MMIO,
# which means every single word access the ARM code makes to the page-data
# buffer (0x400-0x1400) or the spare buffer (0x10020-0x100A0) round-trips
# into Python. Those two ranges are just plain memory from the CPU's point
# of view (no side effects on individual word access), so we map them as
# real Unicorn RAM instead and only sync them to/from Python at page
# load/program time. Only the actual control/status registers (0x1E000+)
# still need a Python callback.
ONENAND_BASE = 0x30000000
BUF_REGION_SIZE = 0x11000  # covers offsets 0x0000-0x11000 (data 0x400-0x1400, spare 0x10020-0x100A0), 4KiB aligned
CTRL_REGION_BASE = ONENAND_BASE + BUF_REGION_SIZE
CTRL_REGION_SIZE = 0x20000 - BUF_REGION_SIZE
DATA_BUF_ADDR = ONENAND_BASE + 0x400
SPARE_BUF_ADDR = ONENAND_BASE + 0x10020
# -----------------------------------------------------------------------

BOOTLOADER_ADDRESES = {
    "C_D1_006 2010/08/18": {
        "FSR_STL_Init": [0x60C090A0, 0x9C],
        "FSR_STL_Open": [0x60C092EC, 0x414],
        "FSR_STL_Read": [0x60C098F8, 0x130],
        "printf_end": 0x60C14930,
        "printf_result": 0x61A46E30,
    },
    "C_D1_001 2011/02/09": {
        "FSR_STL_Init": [0x60C09294, 0x9C],
        "FSR_STL_Open": [0x60C094DC, 0x434],
        "FSR_STL_Read": [0x60C09B48, 0x13C],
        "printf_end": 0x60C1510C,
        "printf_result": 0x61A46E48,
    },
}


def envflag_set(flag):
    return flag in os.environ and os.environ[flag]


def abort(uc):
    print_regs(uc)
    print("!!! ABORT !!!")
    os._exit(-1)


class OnenandFlash:
    """OneNAND image reader with automatic OOB layout detection.

    Supported inputs:
      * data-only .bin with an optional sibling .oob file
      * raw page-interleaved .bin (4096-byte data + 128-byte OOB per page)
      * data-only .bin without OOB (spare bytes are treated as erased, 0xFF)
    """

    def __init__(self, onenand, image_path, pages_per_block):
        self.onenand = onenand
        self.image_path = self._resolve_image_path(image_path)
        self.pages_per_block = pages_per_block
        self.onenand_bin = open(self.image_path, "rb")
        self.onenand_oob = None
        self.oob_stride = 0
        self.layout = None

        raw_size = os.fstat(self.onenand_bin.fileno()).st_size
        page_size = self.onenand.page_size
        spare_size = self.onenand.spare_size
        raw_page_size = page_size + spare_size

        oob_path = self._find_oob_path(self.image_path)
        if oob_path is not None:
            if raw_size == 0 or raw_size % page_size != 0:
                self.close()
                raise ValueError(
                    f"Data image size is not a multiple of {page_size}: {self.image_path}"
                )
            self.layout = "split-oob"
            self.size = raw_size
            self.onenand_oob = open(oob_path, "rb")
            oob_file_size = os.fstat(self.onenand_oob.fileno()).st_size
            page_count = self.size // page_size
            if page_count == 0 or oob_file_size % page_count != 0:
                self.close()
                raise ValueError(f"Cannot determine OOB size from: {oob_path}")
            self.oob_stride = oob_file_size // page_count
            if self.oob_stride == 0:
                self.close()
                raise ValueError(f"OOB file is empty: {oob_path}")
            print(f"OOB detected: {oob_path} ({self.oob_stride} bytes/page)")
        elif (
            raw_size % raw_page_size == 0
            and self._is_power_of_two((raw_size // raw_page_size) * page_size)
        ):
            self.layout = "interleaved-oob"
            page_count = raw_size // raw_page_size
            self.size = page_count * page_size
            print(f"Embedded OOB detected: {self.image_path} ({spare_size} bytes/page)")
        elif raw_size > 0 and raw_size % page_size == 0:
            self.layout = "data-only"
            self.size = raw_size
            print(f"No OOB sidecar found; using data-only image: {self.image_path}")
        else:
            self.close()
            raise ValueError(
                f"Unsupported OneNAND image size ({raw_size} bytes): {self.image_path}"
            )

        self.blocks = self.size // (self.pages_per_block * self.onenand.page_size)
        if self.blocks == 0:
            self.close()
            raise ValueError(f"Image is too small to contain a block: {self.image_path}")

    @staticmethod
    def _is_power_of_two(value):
        return value > 0 and (value & (value - 1)) == 0

    @staticmethod
    def _resolve_image_path(image_path):
        path = Path(image_path).expanduser()
        if path.is_file():
            return path.resolve()
        if path.suffix.lower() != ".bin":
            appended = Path(str(path) + ".bin")
            if appended.is_file():
                return appended.resolve()
        raise FileNotFoundError(f"OneNAND image not found: {image_path}")

    @staticmethod
    def _find_oob_path(image_path):
        candidates = (
            image_path.with_suffix(".oob"),
            Path(str(image_path) + ".oob"),
            image_path.with_suffix(".OOB"),
            Path(str(image_path) + ".OOB"),
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def _read_spare(self, page_index):
        spare_size = self.onenand.spare_size
        if self.layout == "split-oob":
            self.onenand_oob.seek(page_index * self.oob_stride)
            spare = self.onenand_oob.read(self.oob_stride)[:spare_size]
        elif self.layout == "interleaved-oob":
            raw_page_size = self.onenand.page_size + spare_size
            self.onenand_bin.seek(page_index * raw_page_size + self.onenand.page_size)
            spare = self.onenand_bin.read(spare_size)
        else:
            spare = b""
        return bytearray(spare.ljust(spare_size, b"\xFF"))

    def read_page(self, uc, block, page_in_block):
        page_index = block * self.pages_per_block + page_in_block
        off = page_index * self.onenand.page_size
        if off < 0 or off + self.onenand.page_size > self.size:
            raise IndexError(f"OneNAND page out of range: block={block}, page={page_in_block}")

        if self.layout == "interleaved-oob":
            raw_page_size = self.onenand.page_size + self.onenand.spare_size
            self.onenand_bin.seek(page_index * raw_page_size)
        else:
            self.onenand_bin.seek(off)
        data = self.onenand_bin.read(self.onenand.page_size)
        if len(data) != self.onenand.page_size:
            raise EOFError(f"Short read from OneNAND image: {self.image_path}")

        self.onenand.dataram = bytearray(data)
        self.onenand.spareram = self._read_spare(page_index)
        self.onenand._sync_dataram_to_ram(uc)
        self.onenand._sync_spareram_to_ram(uc)

    def close(self):
        for stream_name in ("onenand_bin", "onenand_oob"):
            stream = getattr(self, stream_name, None)
            if stream is not None and not stream.closed:
                stream.close()


class Onenand:
    def __init__(self):
        self.debug = envflag_set("DEBUG_NAND")
        self.debug_trace = self.debug and os.environ["DEBUG_NAND"].lower() == "trace"
        self.int_reads = 0

        self.onenand_mlc = None
        self.onenand_slc = None

        self.syscfg1 = 0xE6C4
        self.mid = 0xEC
        self.did = 0x00
        self.vid = 0x41

        self.page_size = 4096
        self.spare_size = 128

        self.start_addr_1 = 0
        self.start_addr_2 = 0
        self.start_addr_8 = 0

        self.start_buf_reg = 0

        self.pi_mode = False
        self.partition_information = bytearray(b"\xFF" * self.page_size * SLC_PAGES_PER_BLOCK)
        self.int = 0
        self.dataram = bytearray(self.page_size)
        self.spareram = bytearray(self.spare_size)

        self.override_page = dict()
        self.override_spare = dict()

    def _open_image(self, image_path, kind="slc"):
        if kind == "mlc":
            self.print_debug(f"MLC image: '{image_path}'")
            self.onenand_mlc = OnenandFlash(self, image_path, MLC_PAGES_PER_BLOCK)
        else:
            self.print_debug(f"SLC image: '{image_path}'")
            self.onenand_slc = OnenandFlash(self, image_path, SLC_PAGES_PER_BLOCK)

    def open(self, image_path, mlc_image_path=None):
        if mlc_image_path is not None:
            # Explicit SLC + MLC input files (Flex-OneNAND).
            self._open_image(image_path, "slc")
            self._open_image(mlc_image_path, "mlc")
            self.did |= SEPARATION_FLEX << 8
        else:
            # Preserve the convenience of passing just *_slc.bin or *_mlc.bin:
            # if its sibling exists, open both files automatically.
            image = Path(image_path).expanduser()
            if image.suffix.lower() == ".bin":
                stem = image.stem
                ext = image.suffix
            else:
                stem = image.name
                ext = ".bin"
            if stem.endswith(("_slc", "_mlc")):
                base_stem = stem[:-4]
                slc_path = image.with_name(base_stem + "_slc" + ext)
                mlc_path = image.with_name(base_stem + "_mlc" + ext)
                if slc_path.is_file() and mlc_path.is_file():
                    self._open_image(slc_path, "slc")
                    self._open_image(mlc_path, "mlc")
                    self.did |= SEPARATION_FLEX << 8
                else:
                    self._open_image(image_path, "slc")
            else:
                self._open_image(image_path, "slc")

        total_size = 0
        separation = (self.did >> 8) & 0b11
        if self.onenand_slc and self.onenand_mlc:
            assert separation == SEPARATION_FLEX
            separation_str = "Flex-OneNAND"
            total_size = self.onenand_slc.size + self.onenand_mlc.size
            # One SLC block is reserved for Partition Information.
            total_size += SLC_PAGES_PER_BLOCK * self.page_size
        elif self.onenand_slc:
            assert separation == SEPARATION_SLC
            separation_str = "SLC"
            total_size = self.onenand_slc.size
        elif self.onenand_mlc:
            raise ValueError("MLC-only OneNAND images are not supported")
        else:
            raise ValueError("No OneNAND image loaded")

        if total_size == 0:
            raise ValueError("Total NAND size is 0")
        if total_size & (total_size - 1):
            raise ValueError(f"Total NAND size '{total_size}' is not a power of 2")
        density = 0
        while total_size != (1 << (density + 4)) << 20:
            density += 1
            if density > 0xF:
                raise ValueError(
                    f"Total NAND size '{total_size}' doesn't match an expected size"
                )
        self.did |= (density & 0xF) << 4

        self.print_debug("DeviceID: {:04X}".format(self.did))
        self.print_debug("* DeviceID [1:0]  Vcc:           {:0b}".format(self.did & 0b11))
        self.print_debug("* DeviceID [2]    Muxed/Demuxed: {:0b}".format((self.did >> 2) & 0b1))
        self.print_debug("* DeviceID [3]    Single/DDP:    {:0b}".format((self.did >> 3) & 0b1))
        self.print_debug("* DeviceID [7:4]  Density:       {:04b} = {}".format((self.did >> 4) & 0xF, total_size))
        self.print_debug("* DeviceID [9:8]  Separation:    {:02b} = {}".format(separation, separation_str))

        if self.onenand_slc and self.onenand_mlc:
            assert self.onenand_slc.blocks > 0
            alloc = self.onenand_slc.blocks - 1
            if alloc > 0xFFFF:
                self.print_debug("Warning: Alloc value beyond max! wrong files? 0x{:04X}".format(alloc))
                alloc = 0xFFFF
            struct.pack_into("<H", self.partition_information, 0, alloc)

        self.print_debug("* Boundary block: 0x{:04X}".format(self._get_boundary_address()))

    def _get_boundary_address(self):
        # Return the configured boundary block where SLC ends 
        # 0 means only 1'st block is SLC and rest is MLC
        return struct.unpack_from("<H", self.partition_information, 0)[0] & 0xFFF

    def open(self, image_path):
        if image_path.endswith("_slc") or image_path.endswith("_mlc"):
            image_path_base = image_path.removesuffix("slc").removesuffix("mlc")
            image_path_slc = image_path_base + "slc"
            image_path_mlc = image_path_base + "mlc"

            self._open_image(image_path_slc)
            self._open_image(image_path_mlc)

            if self.onenand_slc and self.onenand_mlc:
                # This is a FlexNand dump
                self.did |= SEPARATION_FLEX << 8
        else:
            # SLC
            self._open_image(image_path)

        total_size = 0
        separation = (self.did >> 8) & 0b11
        if self.onenand_slc and self.onenand_mlc:
            assert separation == SEPARATION_FLEX
            separation_str = "Flex-OneNAND"
            total_size = self.onenand_slc.size + self.onenand_mlc.size
            # I assume this is for PI
            total_size += SLC_PAGES_PER_BLOCK * self.page_size
        elif self.onenand_slc:
            assert separation == SEPARATION_SLC
            separation_str = "SLC"
            total_size = self.onenand_slc.size
        elif self.onenand_mlc:
            print("Currently unimplemented OneNand configuration (MLC)")
            os._exit(-1)
        else:
            print("No image loaded?")
            os._exit(-1)

        # Guess density bits value
        # (the formula was revelated to me after having 5h of sleep and 12h train trip the previous day)
        if 0 == total_size:
            print("Total NAND size is 0!")
            os._exit(-1)
        if (total_size & (total_size-1)) != 0:
            print("Total NAND size '{}' is not power of 2!".format(total_size))
            os._exit(-1)
        density = 0
        while total_size != (1 << density + 4) << 20:
            density += 1
            if density > 0xF:
                print("Total NAND size '{}' doesn't match any expected size".format(total_size))
                os._exit(-1)
        self.did |= (density & 0xF) << 4

        self.print_debug("DeviceID: {:04X}".format(self.did))
        self.print_debug("* DeviceID [1:0]  Vcc:           {:0b}".format(self.did & 0b11))
        self.print_debug("* DeviceID [2]    Muxed/Demuxed: {:0b}".format((self.did >> 2) & 0b1))
        self.print_debug("* DeviceID [3]    Single/DDP:    {:0b}".format((self.did >> 3) & 0b1))
        self.print_debug("* DeviceID [7:4]  Density:       {:04b} = {}".format((self.did >> 4) & 0xF, total_size))
        self.print_debug("* DeviceID [9:8]  Separation:    {:02b} = {}".format(separation, separation_str))

        if self.onenand_slc and self.onenand_mlc:
            assert 0 < self.onenand_slc.blocks
            # Program the alloc word with the boundary of SLC / MLC
            alloc = self.onenand_slc.blocks - 1
            if 0xFFFF < alloc:
                self.print_debug("Warning: Alloc value beyond max! wrong files? 0x{:04X}", alloc)
                alloc = 0xFFFF
            struct.pack_into("<H", self.partition_information, 0, alloc)

        self.print_debug("* Boundary block: 0x{:04X}".format(self._get_boundary_address()))

    def print_debug(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    # dataram/spareram are mirrored into real Unicorn RAM (see BUF_REGION
    # constants above) so the ARM code can access them without a Python
    # callback per word. These helpers keep the two in sync at the points
    # where it actually matters (page load / page program), instead of on
    # every single access.
    def _sync_dataram_to_ram(self, uc):
        uc.mem_write(DATA_BUF_ADDR, bytes(self.dataram))

    def _sync_spareram_to_ram(self, uc):
        uc.mem_write(SPARE_BUF_ADDR, bytes(self.spareram))

    def _fetch_dataram_from_ram(self, uc):
        self.dataram = bytearray(uc.mem_read(DATA_BUF_ADDR, self.page_size))

    def _fetch_spareram_from_ram(self, uc):
        self.spareram = bytearray(uc.mem_read(SPARE_BUF_ADDR, self.spare_size))

    def _read(self, uc):
        if self.start_buf_reg != 0x800:
            abort(uc)
        if self.start_addr_8 & 0b11 != 0:
            abort(uc)
        if self.start_addr_2 != 0:
            abort(uc)

        page_in_block = self.start_addr_8 >> 2

        if self.pi_mode:
            # Partition Information only has a single SLC block with no ECC
            if self.start_addr_1 != 0:
                print("Tried to access non 0 block while in PI mode")
                abort(uc)
            if (self.syscfg1 & 0x100) == 0:
                print("PI access must have ECC off")
                abort(uc)
            off = self.page_size * page_in_block
            self.dataram = bytearray(
                self.partition_information[off : off + self.page_size]
            )
            self._sync_dataram_to_ram(uc)
        elif (self.start_addr_1, self.start_addr_8) in self.override_page:
            self.dataram = bytearray(
                self.override_page[(self.start_addr_1, page_in_block)]
            )
            self.spareram = bytearray(
                self.override_spare[(self.start_addr_1, page_in_block)]
            )
            self._sync_dataram_to_ram(uc)
            self._sync_spareram_to_ram(uc)
        elif self.onenand_slc:
            slc_blocks = self.onenand_slc.blocks
            if self.onenand_mlc and slc_blocks <= self.start_addr_1:
                block = self.start_addr_1 - slc_blocks
                self.onenand_mlc.read_page(uc, block, page_in_block)
            else:
                self.onenand_slc.read_page(uc, self.start_addr_1, page_in_block)

    def read_reg(self, uc, offset, size):
        # Note: offset is in the 0x0000-0x20000 OneNAND address-window
        # convention used throughout this file (see onenand_read/onenand_write
        # for how the raw MMIO callback offset is translated into this).
        # The page-data (0x400-0x1400) and spare (0x10020-0x100A0) buffers
        # are backed by real Unicorn RAM now, so they never reach this
        # callback - only the control/status registers below do.
        if self.debug_trace:
            self.print_debug("onenand_read 0x{:X} 0x{:X}".format(offset, size))

        if offset == 0x1E000:
            return self.mid
        elif offset == 0x1E002:
            return self.did
        elif offset == 0x1E004:
            return self.vid
        elif offset == 0x1E442:
            return self.syscfg1
        elif offset == 0x1E482:
            self.int_reads += 1
            if self.int_reads > 100:
                # Pretend its busy until we set flag if its reading INT constantly
                # This seems to be required as in FlexNAND mode the bootloader waits
                # for a while unless its set
                self.int |= 0x8000
            return self.int
        elif offset in [0x1E480, 0x1FE06, 0x1FE04, 0x1FE02, 0x1FE00]:
            return 0
        elif offset == 0x1E49C:
            return 0b100
        else:
            print("onenand_read UNKNOWN offset=0x{:X} size=0x{:X}".format(offset, size))
            abort(uc)

    def write_reg(self, uc, offset, size, value):
        # See the note in read_reg: data/spare buffer offsets never reach
        # this callback anymore, only control/status registers do.
        if self.debug_trace:
            self.print_debug("onenand_write 0x{:X} 0x{:X} 0x{:X}".format(offset, size, value))

        if offset == 0x1E200:
            self.start_addr_1 = value
        elif offset == 0x1E202:
            self.start_addr_2 = value
        elif offset == 0x1E20E:
            self.start_addr_8 = value
        elif offset == 0x1E400:
            self.start_buf_reg = value
        elif offset == 0x1E442:
            self.print_debug("OneNAND SysCfg1: 0x{:04X} -> 0x{:04X}".format(self.syscfg1, value))
            self.syscfg1 = value
        elif offset == 0x1E498:
            pass
        elif offset == 0x1E440:
            # print("==> OneNAND CMD 0x{:X}".format(value))
            self.int = 0
            self.int_reads = 0

            if value == 0x65:
                self.print_debug("OneNAND CMD: OTP read")
                self.int = 0x8000
            elif value == 0x66:
                self.print_debug("OneNAND CMD: PI mode")
                self.pi_mode = True
                self.int = 0x8000
            elif value == 0x00 or value == 0x03:
                self.print_debug("OneNAND CMD: Read A1 0x{:X}, A2 0x{:X}, A8 0x{:X})".format(
                    self.start_addr_1, self.start_addr_2, self.start_addr_8
                ))
                self._read(uc)
                # read completed
                self.int = 0x8080
            elif value == 0xF0:
                self.print_debug("OneNAND CMD: Reset flash core")
                if self.pi_mode:
                    self.print_debug("Leaving PI mode")
                    self.pi_mode = False
                self.int = 0x8010
            elif value == 0x23:
                self.print_debug("OneNAND CMD: Unlock NAND array a block")
                self.int = 0x8004
            elif value == 0x94:
                # Erase block
                self.print_debug("OneNAND CMD: Erase block 0x{:X}".format(self.start_addr_1))
                if self.pi_mode:
                    self.partition_information = bytearray(b"\xff" * self.page_size)
                else:
                    # Use MLC amount since is bigger
                    for x in range(MLC_PAGES_PER_BLOCK):
                        self.override_page[(self.start_addr_1, x)] = b"\xFF" * self.page_size
                        self.override_spare[(self.start_addr_1, x)] = b"\xFF" * self.spare_size
                self.int = 0xFFFF
            elif value == 0x80:
                # Program page
                page_in_block = self.start_addr_8 >> 2
                self.print_debug("OneNAND CMD: Program page 0x{:X}:0x{:X}".format(
                    self.start_addr_1, page_in_block
                ))
                assert self.start_addr_8 % 4 == 0
                # dataram lives in Unicorn RAM now; pull the page the ARM
                # code just wrote into it before consuming it.
                self._fetch_dataram_from_ram(uc)
                if self.pi_mode:
                    off = page_in_block * self.page_size
                    self.partition_information = (
                        self.partition_information[:off]
                        + self.dataram
                        + self.partition_information[off + self.page_size :]
                    )
                    assert len(self.partition_information) == (self.page_size * SLC_PAGES_PER_BLOCK)
                else:
                    self._fetch_spareram_from_ram(uc)
                    self.override_page[(self.start_addr_1, page_in_block)] = self.dataram
                    self.override_spare[(self.start_addr_1, page_in_block)] = self.spareram
                self.int = 0xFFFF
            else:
                print("onenand_write UNKNOWN CMD REG value=0x{:X}".format(value))
                abort(uc)
        else:
            print("onenand_write UNKNOWN offset=0x{:X} size=0x{:X}".format(offset, size))
            abort(uc)


onenand = Onenand()


def onenand_read(uc, offset, size, data):
    # print("onenand read 0x{:08X} size 0x{:X}".format(offset, size))
    # This callback only sees the control/status register window, mapped at
    # CTRL_REGION_BASE, so `offset` is relative to that. read_reg/write_reg
    # use offsets relative to ONENAND_BASE (0x30000000), so shift back.
    return onenand.read_reg(uc, offset + BUF_REGION_SIZE, size)


def onenand_write(uc, offset, size, value, data):
    # print("onenand write 0x{:08X} size 0x{:X} data 0x{:X}".format(offset, size, value))
    onenand.write_reg(uc, offset + BUF_REGION_SIZE, size, value)


def print_regs(uc):
    regs = [
        ("r0", UC_ARM_REG_R0),
        ("r1", UC_ARM_REG_R1),
        ("r2", UC_ARM_REG_R2),
        ("r3", UC_ARM_REG_R3),
        ("r4", UC_ARM_REG_R4),
        ("r5", UC_ARM_REG_R5),
        ("r6", UC_ARM_REG_R6),
        ("r7", UC_ARM_REG_R7),
        ("r8", UC_ARM_REG_R8),
        ("r9", UC_ARM_REG_R9),
        ("r10", UC_ARM_REG_R10),
        ("r11", UC_ARM_REG_R11),
        ("r12", UC_ARM_REG_R12),
        ("lr", UC_ARM_REG_LR),
        ("pc", UC_ARM_REG_PC),
        ("sp", UC_ARM_REG_SP),
    ]

    for name, reg in regs:
        print(">>> {} : 0x{:08X}".format(name, uc.reg_read(reg)))


def post_print(uc, address, size, user_data):
    data = uc.mem_read(BOOTLOADER_ADDRESES[user_data]["printf_result"], 0x2000)
    data = data[: data.find(b"\x00")]
    print("!! PRINT !! {}".format(data.decode("ascii").rstrip()))


def emu_run_func(uc, bootloader, func):
    funcdata = BOOTLOADER_ADDRESES[bootloader][func]
    uc.emu_start(funcdata[0], funcdata[0] + funcdata[1])


def main():
    script_dir = Path(__file__).resolve().parent
    argv = sys.argv[1:]
    bootloader_arg = None
    if "--bootloader" in argv or "-b" in argv:
        flag = "--bootloader" if "--bootloader" in argv else "-b"
        flag_index = argv.index(flag)
        if flag_index + 1 >= len(argv):
            raise SystemExit(f"{flag} requires a file path")
        bootloader_arg = argv[flag_index + 1]
        del argv[flag_index:flag_index + 2]

    if len(argv) not in (2, 3):
        raise SystemExit(
            "Usage:\n"
            "  python emu.py path/to/onenand.bin path/to/output\n"
            "  python emu.py path/to/onenand_slc.bin path/to/onenand_mlc.bin path/to/output\n"
            "Optional: --bootloader path/to/secondbl.bin"
        )

    if len(argv) == 2:
        image_paths = [argv[0]]
        output_path = Path(argv[1]).expanduser().resolve()
    else:
        image_paths = [argv[0], argv[1]]
        output_path = Path(argv[2]).expanduser().resolve()

    if bootloader_arg:
        bootloader_path = Path(bootloader_arg).expanduser().resolve()
    else:
        bootloader_path = script_dir / "secondbl.bin"
        if not bootloader_path.is_file() and (Path.cwd() / "secondbl.bin").is_file():
            bootloader_path = Path.cwd() / "secondbl.bin"

    if not bootloader_path.is_file():
        raise FileNotFoundError(
            f"Bootloader file not found: {bootloader_path} (use --bootloader to specify it)"
        )

    with open(bootloader_path, "rb") as inf:
        bootloader_code = inf.read()
    if len(bootloader_code) < 0x1F3:
        raise ValueError(f"Bootloader is too small: {bootloader_path}")

    bootloader = bootloader_code[0x1E0:0x1F3].decode("ascii").replace("\0", " ")
    print("Emulated BL '{:}' string: '{:}'".format(bootloader_path, bootloader))
    if bootloader not in BOOTLOADER_ADDRESES:
        raise ValueError(f"Unknown bootloader provided: '{bootloader}'")

    global onenand
    onenand = Onenand()
    if len(image_paths) == 2:
        onenand.open(image_paths[0], image_paths[1])
    else:
        onenand.open(image_paths[0])

    # Unicorn writes the generated image beside this script first. Once the
    # conversion finishes, move that temporary file to the requested target.
    fd, temp_output = tempfile.mkstemp(prefix=".emu_part_1A_", suffix=".bin", dir=script_dir)
    os.close(fd)
    temp_output_path = Path(temp_output)
    try:
        partnum = 0x1A
        uc = Uc(UC_ARCH_ARM, UC_MODE_ARM)

        uc.mem_map(0x60C01000, 0x40000)
        uc.mem_write(0x60C01000, bootloader_code)

        uc.reg_write(UC_ARM_REG_SP, 0xE7A91FFC)
        uc.mem_map(0xE7A90000, 0x2000)
        uc.mem_map(0x61100000, 0x950000)
        uc.mem_map(0x70000000, 0x100000)
        info_ptr = 0x70000000 + 0x4000
        buf_ptr = 0x70000000

        # Map data/spare buffers as RAM and only control registers as MMIO.
        uc.mem_map(ONENAND_BASE, BUF_REGION_SIZE)
        uc.mmio_map(
            CTRL_REGION_BASE, CTRL_REGION_SIZE, onenand_read, None, onenand_write, None
        )

        printf_end = BOOTLOADER_ADDRESES[bootloader]["printf_end"]
        uc.hook_add(
            UC_HOOK_CODE,
            post_print,
            begin=printf_end,
            end=printf_end,
            user_data=bootloader,
        )

        try:
            emu_run_func(uc, bootloader, "FSR_STL_Init")
            result = uc.reg_read(UC_ARM_REG_R0)
            print("FSR_STL_Init => 0x{:X}".format(result))
            if result != 0:
                raise RuntimeError(f"FSR_STL_Init failed: 0x{result:X}")

            uc.reg_write(UC_ARM_REG_R0, 0)
            uc.reg_write(UC_ARM_REG_R1, partnum)
            uc.reg_write(UC_ARM_REG_R2, info_ptr)
            uc.reg_write(UC_ARM_REG_R3, 0)
            emu_run_func(uc, bootloader, "FSR_STL_Open")
            result = uc.reg_read(UC_ARM_REG_R0)
            print("FSR_STL_Open => 0x{:X}".format(result))
            if result != 0:
                raise RuntimeError(f"FSR_STL_Open failed for part 0x1A: 0x{result:X}")

            print("Read start")
            blk = 0
            with open(temp_output_path, "wb") as outf:
                while True:
                    uc.reg_write(UC_ARM_REG_SP, 0x70010000)
                    uc.reg_write(UC_ARM_REG_R0, 0)
                    uc.reg_write(UC_ARM_REG_R1, partnum)
                    uc.reg_write(UC_ARM_REG_R2, blk)
                    uc.reg_write(UC_ARM_REG_R3, 1)
                    uc.mem_write(
                        uc.reg_read(UC_ARM_REG_SP), struct.pack("<II", buf_ptr, 0)
                    )
                    if blk % 0x1000 == 0:
                        print("Reading part 0x{:X} block 0x{:X}".format(partnum, blk))
                    emu_run_func(uc, bootloader, "FSR_STL_Read")
                    ret = uc.reg_read(UC_ARM_REG_R0)
                    if ret == 0x80030000:
                        break
                    if ret != 0:
                        raise RuntimeError(f"FSR_STL_Read failed: 0x{ret:X}")
                    outf.write(uc.mem_read(buf_ptr, 512))
                    blk += 1

            if blk == 0:
                raise RuntimeError("No blocks were read for part 0x1A")
        except unicorn.UcError:
            print_regs(uc)
            raise

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temp_output_path), str(output_path))
        temp_output_path = None
        print(f"Output written: {output_path}")
    finally:
        if temp_output_path is not None and temp_output_path.exists():
            temp_output_path.unlink()
        for flash in (onenand.onenand_slc, onenand.onenand_mlc):
            if flash is not None:
                flash.close()


if __name__ == "__main__":
    main()

# sp = E7A91FFC
