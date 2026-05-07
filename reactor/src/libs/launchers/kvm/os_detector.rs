//! Detects whether a libvirt guest is running Windows or Linux based on its disk image.
//!
//! Detection proceeds in two stages: a fast XML-based heuristic check followed by a
//! disk-based inspection that uses `qemu-img` to convert the qcow2 image to raw and then
//! sniffs partition tables and filesystem magic bytes.
//!
//! Results are cached using `quick_cache` keyed on the disk's path, size, and modification
//! time. When many workers concurrently inspect the same image, the cache coalesces the
//! work so that only one worker runs detection while the others wait for its result.

use quick_cache::sync::Cache;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::io::{AsyncReadExt, AsyncSeekExt, SeekFrom};
use tracing::{Level, event, instrument};

use crate::Error;

/// The detected operating system family of a guest
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, strum::Display)]
pub enum GuestOs {
    /// A Linux guest, indicated by an ext/XFS/btrfs/swap filesystem, LUKS volume, or LVM PV
    Linux,
    /// A Windows guest, indicated by an NTFS or ReFS filesystem
    Windows,
    /// Inspection completed successfully but no Windows or Linux signal was found
    Unknown,
}

/// The cache key used to look up a previously-detected OS
///
/// Includes the image's size and mtime so that replacing the image at the same path
/// invalidates the cached result automatically without any explicit cache management.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    /// The absolute path to the disk image
    path: PathBuf,
    /// The size of the disk image in bytes
    size: u64,
    /// The last modification time of the disk image
    mtime: SystemTime,
}

/// Detects the operating system of a libvirt guest from its XML definition and disk image.
///
/// Holds an internal cache shared across all clones of the detector via `Arc`, so cloning
/// is cheap and clones share cache state. Detection results are coalesced across concurrent
/// callers: if multiple workers request detection for the same uncached image at the same
/// time, only one runs the actual detection while the others await its result.
#[derive(Debug, Clone)]
pub struct OsDetector {
    /// The cache of previously-detected OS results keyed by image content fingerprint
    cache: Arc<Cache<CacheKey, GuestOs>>,
}

/// The size of each disk sector in bytes (standard for x86 disks)
const SECTOR_SIZE: u64 = 512;

/// The number of bytes to read from the start of each partition for filesystem detection.
///
/// Sized to comfortably include the btrfs superblock at offset 0x10000 (65536) plus the
/// 4 KiB superblock itself, so a single read covers all filesystems we look for.
const PARTITION_READ_BYTES: usize = 0x11000;

/// The qcow2 magic bytes that appear at the start of every qcow2 file
const QCOW2_MAGIC: &[u8; 4] = b"QFI\xfb";

impl OsDetector {
    /// Create a new detector with the given cache capacity
    ///
    /// # Arguments
    ///
    /// * `capacity` - The max capacity for the cache
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Arc::new(Cache::new(capacity)),
        }
    }

    /// Detect the OS of a guest, using the cache when possible
    ///
    /// Inspects a VM disk image by converting the qcow2 image to raw temporarily,
    /// inspecting partitions for filesystem magic numbers and assuming OS based on the
    /// filesystem. Caches results and coalesces concurrent first-time detections.
    ///
    /// # Arguments
    ///
    /// * `temp_dir` - The directory to store temporary raw VM images for introspection
    /// * `disk_path` - The path to the guest's primary qcow2 disk image
    #[instrument(name = "OsDetector::detect", skip_all, fields(disk = %disk_path.display()), err(Debug))]
    pub async fn detect(&self, temp_dir: &Path, disk_path: &Path) -> Result<GuestOs, Error> {
        // build the content-aware cache key from a stat() of the disk
        let metadata = tokio::fs::metadata(disk_path).await?;
        let key = CacheKey {
            path: disk_path.to_path_buf(),
            size: metadata.len(),
            mtime: metadata.modified()?,
        };
        // get_or_insert_async coalesces concurrent misses on the same key, so if many workers
        // simultaneously hit a cache miss for the same image, only one runs the detection
        let owned_path = disk_path.to_path_buf();
        let os = self
            .cache
            .get_or_insert_async(&key, async move {
                event!(Level::INFO, "Cache Miss");
                detect_from_disk(temp_dir, &owned_path).await
            })
            .await?;
        Ok(os)
    }
}

/// Detect the OS by inspecting the disk image's partition table and filesystem signatures
///
/// Converts the qcow2 image to a raw temporary file via `qemu-img`, then walks the
/// partitions looking for filesystem magic bytes. Returns `Unknown` if the disk is
/// readable but has no recognizable Windows or Linux signal.
///
/// # Arguments
///
/// * `disk_path` - The path to the guest's qcow2 disk image
#[instrument(name = "detect_from_disk", skip(temp_dir))]
async fn detect_from_disk(temp_dir: &Path, disk_path: &Path) -> Result<GuestOs, Error> {
    // verify the file actually starts with the qcow2 magic before invoking qemu-img
    verify_qcow2_magic(disk_path).await?;
    // create a temp file that will be cleaned up on drop, even on error or panic
    let raw_file = tempfile::NamedTempFile::new_in(temp_dir)?;
    let raw_path = raw_file.path().to_path_buf();
    // convert qcow2 to raw so we can read it with simple positioned I/O
    convert_qcow2_to_raw(disk_path, &raw_path).await?;
    // inspect the raw image for partition + filesystem signals
    inspect_raw_image(&raw_path).await
}

/// Read the first four bytes of `disk_path` and verify they are the qcow2 magic
///
/// # Arguments
///
/// * `disk_path` - The path to the disk image to verify
async fn verify_qcow2_magic(disk_path: &Path) -> Result<(), Error> {
    let mut file = tokio::fs::File::open(disk_path).await?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).await?;
    if &magic != QCOW2_MAGIC {
        return Err(Error::OsDetect(format!(
            "Disk image '{}' is not a qcow2 image (bad magic: {:02x?})",
            disk_path.display(),
            magic
        )));
    }
    Ok(())
}

/// Convert a qcow2 image to a raw image at `raw_path` using `qemu-img`
///
/// # Arguments
///
/// * `qcow2_path` - The path to the source qcow2 image
/// * `raw_path` - The path at which to write the raw output image
async fn convert_qcow2_to_raw(qcow2_path: &Path, raw_path: &Path) -> Result<(), Error> {
    let output = tokio::process::Command::new("qemu-img")
        .args(["convert", "-O", "raw"])
        .arg(qcow2_path)
        .arg(raw_path)
        .output()
        .await?;
    if !output.status.success() {
        return Err(Error::OsDetect(format!(
            "qemu-img convert of '{}' failed: {}",
            qcow2_path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(())
}

/// Walk the partitions in a raw disk image looking for OS signals
///
/// First checks the disk itself (offset 0) for partitionless layouts (common in cloud
/// images), then parses GPT and MBR partition tables and inspects the start of each
/// partition for filesystem magic bytes. Returns the first definitive signal found,
/// or `Unknown` if nothing recognizable is present.
///
/// # Arguments
///
/// * `raw_path` - The path to the raw disk image to inspect
#[instrument(name = "os_detector::inspect_raw_image")]
async fn inspect_raw_image(raw_path: &Path) -> Result<GuestOs, Error> {
    let mut file = tokio::fs::File::open(raw_path).await?;
    // some images put a filesystem directly on the disk with no partition table
    let head = read_at(&mut file, 0, PARTITION_READ_BYTES).await?;
    if let Some(os) = identify_filesystem(&head)? {
        return Ok(os);
    }
    // try GPT first since it supersedes MBR; protective MBR will look invalid for fs sniffing
    if let Some(os) = scan_gpt_partitions(&mut file).await? {
        return Ok(os);
    }
    // fall back to MBR for older images
    if let Some(os) = scan_mbr_partitions(&mut file, &head).await? {
        return Ok(os);
    }
    event!(
        Level::WARN,
        msg = "Unable to detect OS of VM image",
        image_path = format!("{}", raw_path.display())
    );
    Ok(GuestOs::Unknown)
}

/// Read `len` bytes starting at `offset` from `file` into a fresh `Vec`
///
/// Reads up to `len` bytes; if the file is shorter than `offset + len`, returns
/// only what was actually read rather than failing. This matters for tiny images
/// or partitions near the end of the disk.
///
/// # Arguments
///
/// * `file` - The async file handle to read from
/// * `offset` - The byte offset at which to begin reading
/// * `len` - The maximum number of bytes to read
#[instrument(name = "os_detector::read_at", skip(file))]
async fn read_at(file: &mut tokio::fs::File, offset: u64, len: usize) -> Result<Vec<u8>, Error> {
    file.seek(SeekFrom::Start(offset)).await?;
    let mut buf = vec![0u8; len];
    let mut total = 0;
    // read_exact would error on short reads; loop with read() so partial reads are ok
    while total < len {
        let n = file.read(&mut buf[total..]).await?;
        if n == 0 {
            break;
        }
        total += n;
    }
    buf.truncate(total);
    Ok(buf)
}

/// Scan GPT partitions on `file` for filesystem signals
///
/// Returns `Ok(None)` if no GPT is present (signature mismatch). Returns `Ok(Some(os))`
/// as soon as any partition yields a definitive signal.
///
/// # Arguments
///
/// * `file` - The async file handle to read from
#[instrument(name = "os_detector::scan_gpt_partitions", skip_all)]
async fn scan_gpt_partitions(file: &mut tokio::fs::File) -> Result<Option<GuestOs>, Error> {
    // GPT header lives in LBA 1 (offset 512); first 8 bytes are the signature "EFI PART"
    let header = read_at(
        file,
        SECTOR_SIZE,
        SECTOR_SIZE.try_into().map_err(|_| {
            Error::new(format!("SECTOR SIZE '{SECTOR_SIZE}' does not fit in usize"))
        })?,
    )
    .await?;
    if header.len() < 92 || &header[0..8] != b"EFI PART" {
        return Ok(None);
    }
    // partition entry array LBA at offset 72 (u64 LE), entries are 128 bytes each
    let part_lba = u64::from_le_bytes(
        header[72..80]
            .try_into()
            .map_err(|err| Error::with_context("Invalid LBA slice", err))?,
    );
    let num_entries = u32::from_le_bytes(
        header[80..84]
            .try_into()
            .map_err(|err| Error::with_context("Invalid num entries slice", err))?,
    );
    let entry_size = u32::from_le_bytes(
        header[84..88]
            .try_into()
            .map_err(|err| Error::with_context("Invalid entry size slice", err))?,
    );
    // sanity-check the values to avoid pathological reads on a corrupt or malicious header
    if !(128..=4096).contains(&entry_size) || num_entries == 0 || num_entries > 1024 {
        return Ok(None);
    }
    // each entry: starting LBA at offset 32 (u64 LE), ending LBA at 40, all-zero LBAs = unused
    let array_offset = part_lba.saturating_mul(SECTOR_SIZE);
    let array_size = u64::from(num_entries)
        .saturating_mul(u64::from(entry_size))
        .try_into()
        .map_err(|_| Error::new("num_entries * entry_size does not fit in usize"))?;
    let array = read_at(file, array_offset, array_size).await?;
    for i in 0..num_entries as usize {
        let off = i * entry_size as usize;
        if off + 48 > array.len() {
            break;
        }
        let entry = &array[off..off + 48];
        let start_lba = u64::from_le_bytes(
            entry[32..40]
                .try_into()
                .map_err(|err| Error::with_context("Invalid start LBA slice", err))?,
        );
        let end_lba = u64::from_le_bytes(
            entry[40..48]
                .try_into()
                .map_err(|err| Error::with_context("Invalid end LBA slice", err))?,
        );
        // skip empty entries
        if start_lba == 0 && end_lba == 0 {
            continue;
        }
        let part_offset = start_lba.saturating_mul(SECTOR_SIZE);
        let data = read_at(file, part_offset, PARTITION_READ_BYTES).await?;
        if let Some(os) = identify_filesystem(&data)? {
            return Ok(Some(os));
        }
    }
    Ok(None)
}

/// Scan MBR partitions on `file` for filesystem signals
///
/// `boot_sector` should be the first sector already read from the disk so we can avoid
/// re-reading it. Returns `Ok(None)` if no valid MBR is present.
///
/// # Arguments
///
/// * `file` - The async file handle to read from
/// * `boot_sector` - The first sector of the disk, already read into memory
#[instrument(name = "os_detector::scan_mbr_partitions", skip_all)]
async fn scan_mbr_partitions(
    file: &mut tokio::fs::File,
    boot_sector: &[u8],
) -> Result<Option<GuestOs>, Error> {
    // MBR signature 0x55AA at offsets 510-511
    if boot_sector.len() < 512 || boot_sector[510] != 0x55 || boot_sector[511] != 0xAA {
        return Ok(None);
    }
    // partition table is 4 entries of 16 bytes each starting at offset 446
    for i in 0..4 {
        let off = 446 + i * 16;
        let entry = &boot_sector[off..off + 16];
        let part_type = entry[4];
        // type 0x00 means the entry is unused
        if part_type == 0 {
            continue;
        }
        // starting LBA at offset 8 (u32 LE), sector count at offset 12 (also u32 LE)
        let start_lba = u32::from_le_bytes(entry[8..12].try_into().unwrap());
        let sectors = u32::from_le_bytes(entry[12..16].try_into().unwrap());
        if start_lba == 0 || sectors == 0 {
            continue;
        }
        // type 0xEE is the GPT protective partition; skip it - we'd have used GPT already
        if part_type == 0xEE {
            continue;
        }
        let part_offset = u64::from(start_lba) * SECTOR_SIZE;
        let data = read_at(file, part_offset, PARTITION_READ_BYTES).await?;
        if let Some(os) = identify_filesystem(&data)? {
            return Ok(Some(os));
        }
    }
    Ok(None)
}

/// Examine the start of a partition for known filesystem magic bytes
///
/// Returns `Some(os)` for any recognized Windows or Linux filesystem signal, or `None`
/// if the bytes don't match anything we know about (e.g. FAT, which is ambiguous and
/// commonly appears as the EFI System Partition on both Windows and Linux guests).
///
/// # Arguments
///
/// * `data` - The first bytes of a partition (or the whole disk for unpartitioned images)
#[instrument(name = "os_detector::identify_filesystem", skip_all)]
fn identify_filesystem(data: &[u8]) -> Result<Option<GuestOs>, Error> {
    // NTFS: OEM ID "NTFS    " (4 chars + 4 spaces) at offset 3
    if data.len() >= 11 && &data[3..11] == b"NTFS    " {
        return Ok(Some(GuestOs::Windows));
    }
    // ReFS: OEM ID "ReFS\0\0\0\0" at offset 3, plus "FSRS" marker at offset 0x10
    if data.len() >= 0x14
        && &data[3..7] == b"ReFS"
        && &data[7..11] == b"\0\0\0\0"
        && &data[0x10..0x14] == b"FSRS"
    {
        return Ok(Some(GuestOs::Windows));
    }
    // ext2/3/4: superblock at offset 1024, magic 0xEF53 (LE) at superblock offset 0x38
    if data.len() >= 0x43A && data[0x438] == 0x53 && data[0x439] == 0xEF {
        return Ok(Some(GuestOs::Linux));
    }
    // XFS: magic "XFSB" (big-endian 0x58465342) at offset 0
    if data.len() >= 4 && &data[0..4] == b"XFSB" {
        return Ok(Some(GuestOs::Linux));
    }
    // btrfs: superblock at offset 0x10000, magic "_BHRfS_M" at superblock offset 0x40
    if data.len() >= 0x10048 && &data[0x10040..0x10048] == b"_BHRfS_M" {
        return Ok(Some(GuestOs::Linux));
    }
    // LUKS1 / LUKS2 primary header: magic "LUKS\xba\xbe" at offset 0
    if data.len() >= 6 && &data[0..6] == b"LUKS\xba\xbe" {
        return Ok(Some(GuestOs::Linux));
    }
    // Linux swap: 10-byte signature "SWAPSPACE2" at the end of the first 4 KiB page
    // older "SWAP-SPACE" with hyphen also exists; check both
    if data.len() >= 4096 {
        let tail = &data[4086..4096];
        if tail == b"SWAPSPACE2" || tail == b"SWAP-SPACE" {
            return Ok(Some(GuestOs::Linux));
        }
    }
    // LVM2 PV: "LABELONE" can appear in any of the first 4 sectors (offsets 0, 512, 1024, 1536)
    // followed shortly after by "LVM2 001" type indicator
    for sector in 0..4u64 {
        let base = (sector * SECTOR_SIZE)
            .try_into()
            .map_err(|_| Error::new("sector * SECTOR_SIZE does not fit in usize"))?;
        if data.len() < base + 0x20 {
            break;
        }
        if &data[base..base + 8] == b"LABELONE" && &data[base + 0x18..base + 0x20] == b"LVM2 001" {
            return Ok(Some(GuestOs::Linux));
        }
    }
    Ok(None)
}

#[allow(clippy::expect_used, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic partition prefix containing the given OEM ID at offset 3.
    ///
    /// # Arguments
    ///
    /// * `oem` - The 8-byte OEM ID to embed
    fn ntfs_like(oem: &[u8; 8]) -> Vec<u8> {
        let mut v = vec![0u8; 512];
        v[0..3].copy_from_slice(&[0xEB, 0x52, 0x90]);
        v[3..11].copy_from_slice(oem);
        v[510] = 0x55;
        v[511] = 0xAA;
        v
    }

    #[test]
    fn identifies_ntfs() {
        let data = ntfs_like(b"NTFS    ");
        assert_eq!(identify_filesystem(&data).unwrap(), Some(GuestOs::Windows));
    }

    #[test]
    fn identifies_refs() {
        let mut data = vec![0u8; 512];
        data[3..7].copy_from_slice(b"ReFS");
        data[0x10..0x14].copy_from_slice(b"FSRS");
        assert_eq!(identify_filesystem(&data).unwrap(), Some(GuestOs::Windows));
    }

    #[test]
    fn identifies_ext4() {
        let mut data = vec![0u8; 0x500];
        // 1024 bytes of padding, then superblock with magic at offset 0x38
        data[0x438] = 0x53;
        data[0x439] = 0xEF;
        assert_eq!(identify_filesystem(&data).unwrap(), Some(GuestOs::Linux));
    }

    #[test]
    fn identifies_xfs() {
        let mut data = vec![0u8; 512];
        data[0..4].copy_from_slice(b"XFSB");
        assert_eq!(identify_filesystem(&data).unwrap(), Some(GuestOs::Linux));
    }

    #[test]
    fn identifies_btrfs() {
        let mut data = vec![0u8; 0x11000];
        data[0x10040..0x10048].copy_from_slice(b"_BHRfS_M");
        assert_eq!(identify_filesystem(&data).unwrap(), Some(GuestOs::Linux));
    }

    #[test]
    fn identifies_luks() {
        let mut data = vec![0u8; 512];
        data[0..6].copy_from_slice(b"LUKS\xba\xbe");
        assert_eq!(identify_filesystem(&data).unwrap(), Some(GuestOs::Linux));
    }

    #[test]
    fn identifies_swap() {
        let mut data = vec![0u8; 4096];
        data[4086..4096].copy_from_slice(b"SWAPSPACE2");
        assert_eq!(identify_filesystem(&data).unwrap(), Some(GuestOs::Linux));
    }

    #[test]
    fn identifies_lvm() {
        let mut data = vec![0u8; 1024];
        // LABELONE at sector 1 (offset 512) is the most common location
        data[512..520].copy_from_slice(b"LABELONE");
        data[512 + 0x18..512 + 0x20].copy_from_slice(b"LVM2 001");
        assert_eq!(identify_filesystem(&data).unwrap(), Some(GuestOs::Linux));
    }

    #[test]
    fn identifies_fat_as_unknown() {
        // FAT32 OEM ID "MSDOS5.0" should not match any signal
        let data = ntfs_like(b"MSDOS5.0");
        assert_eq!(identify_filesystem(&data).unwrap(), None);
    }

    #[test]
    fn empty_data_returns_none() {
        assert_eq!(identify_filesystem(&[]).unwrap(), None);
    }

    #[tokio::test]
    #[ignore = "requires qemu-img and a test qcow2 image to be present"]
    async fn integration_detect_from_disk() {
        // To run: cargo test -- --ignored
        // Set TEST_QCOW2 to the path of a qcow2 image and TEST_EXPECTED to "linux" or "windows".
        let path = std::env::var("TEST_QCOW2").expect("TEST_QCOW2 not set");
        let expected = std::env::var("TEST_EXPECTED").expect("TEST_EXPECTED not set");
        let detector = OsDetector::new(100);
        // write the temporary file to the system temporary directory
        let temp_dir = std::env::temp_dir();
        let os = detector
            // what temp dir path here...?
            .detect(&temp_dir, Path::new(&path))
            .await
            .expect("detection failed");
        match expected.as_str() {
            "linux" => assert_eq!(os, GuestOs::Linux),
            "windows" => assert_eq!(os, GuestOs::Windows),
            other => panic!("unknown expected value: {other}"),
        }
    }
}
