//! The footers for the cart format

use crate::Error;
use std::io::Write;

/// The length of a standard `CaRT` footer
pub const FOOTER_LEN: usize = 28;
/// The magic number beginning the footer
pub static MAGIC_NUM: &[u8; 4] = b"TRAC";

/// The mandatory footer object for `CaRT`
///
/// While the [cart docs](https://bitbucket.org/cse-assemblyline/cart/src/master/) say this footer is 32 bytes long in reality it is only 28 bytes. It
/// appears the python cart implementation adds 4 bytes to the end of the stream. This can be
/// seen by carting a file with the python implementation and then running hexdump on the output.
///
/// ```text
/// (venv) ➜ cart git:(master) ✗ cart corn
/// (venv) ➜ cart git:(master) ✗ hd corn.cart -v
/// 00000000  43 41 52 54 01 00 00 00  00 00 00 00 00 00 03 01  |CART............|
/// 00000010  04 01 05 09 02 06 03 01  04 01 05 09 02 06 0f 00  |................|
/// 00000020  00 00 00 00 00 00 c2 a4  a5 5c 53 d5 43 f7 79 76  |.........\S.C.yv|
/// 00000030  39 d6 6f 11 9d c1 87 b8  10 f5 7c 10 03 74 df b5  |9.o.......|..t..|
/// 00000040  a6 01 23 34 a6 92 c2 a4  a7 58 50 d7 15 a5 79 2f  |..#4.....XP...y/|
/// 00000050  74 9d 23 1f c2 c8 db d3  8b 07 b0 7b 1a 22 79 4a  |t.#........{."yJ|
/// 00000060  0b 8e e2 b9 60 73 74 2b  56 c6 71 67 62 e9 1d 47  |....`st+V.qgb..G|
/// 00000070  3b 28 d8 87 f1 d4 50 17  5c 42 e0 33 3d da e5 07  |;(....P.\B.3=...|
/// 00000080  09 28 ec 4f f0 10 83 58  d3 d7 ef f2 48 fa a4 bd  |.(.O...X....H...|
/// 00000090  87 e4 5e 51 7e e4 d8 6f  66 0b 32 db 69 ef c8 74  |..^Q~..of.2.i..t|
/// 000000a0  c9 ed d1 78 a8 88 07 45  09 13 55 88 b8 48 ee 52  |...x...E..U..H.R|
/// 000000b0  1c ec f4 2e 6b 79 0a 97  0b 75 b3 96 76 21 83 2c  |....ky...u..v!.,|
/// 000000c0  14 a5 e2 17 03 99 d6 82  da 8a 03 e4 32 fe 62 eb  |............2.b.|
/// 000000d0  29 67 b6 95 eb 9c 69 21  8a d3 e0 97 af c4 22 da  |)g....i!......".|
/// 000000e0  ef 58 29 58 4c 8f d7 e0  35 f1 34 7d 55 7a f1 c6  |.X)XL...5.4}Uz..|
/// 000000f0  f6 c7 3d e2 60 d3 8c 14  64 58 d1 54 52 41 43 00  |..=.`...dX.TRAC.|
/// 00000100  00 00 00 00 00 00 00 46  00 00 00 00 00 00 00 b5  |.......F........|
/// 00000110  00 00 00 00 00 00 00                              |.......|
/// 00000117
/// ```
///
/// Currently this footer is not used as cart-rs does not support an optional footer (it will just
/// be ignored when uncarting a file).
#[derive(Debug, Clone)]
pub struct Footer {
    // The size of the optional footer to skip
    pub opt_len: u64,
}

impl Footer {
    /// Build the footer for this carted file
    #[must_use]
    pub fn new_buffer() -> [u8; FOOTER_LEN] {
        // build our mandatory footer vector of 28 bytes
        let mut footer = [0; FOOTER_LEN];
        // write the CaRT magic number
        Self::write_magic_num(&mut footer);
        footer
    }

    /// Write the `CaRT` footer to a buffer
    ///
    /// # Arguments
    ///
    /// * `buf` - The buffer to write the footer to
    pub fn write(mut buf: &mut [u8]) -> Result<(), std::io::Error> {
        let mut footer: [u8; FOOTER_LEN] = [0; FOOTER_LEN];
        Self::write_magic_num(&mut footer);
        buf.write_all(&footer)
    }

    /// Write the `CaRT` magic number to the beginning of a buffer
    ///
    /// # Arguments
    ///
    /// * `buf` - The buffer to write the `CaRT` magic number to
    fn write_magic_num(buf: &mut [u8]) {
        buf[..MAGIC_NUM.len()].copy_from_slice(MAGIC_NUM);
    }

    /// Gets the footer from the last 28 bytes of the raw binary
    ///
    /// # Arguments
    ///
    /// * `raw` - The last 28 bytes of the binary containing the footer
    ///
    /// # Errors
    ///
    /// If this buffer does not start with the TRAC magic number then an error will be returned.
    ///
    /// # Errors
    ///
    /// If any IO errors occur then an error will be returned and the header will fail to write.
    /// IO errors should only happen if an insufficient buffer is provided.
    pub fn get(raw: &[u8]) -> Result<Self, Error> {
        // get the last 28 bytes of this buffer
        let end = &raw[raw.len() - 28..];
        // make sure the magic numbers match carts magic number
        if end[..4] == *MAGIC_NUM {
            // setup a bincode config
            let config = bincode::config::standard();
            // extract the length of the optional footer
            let (opt_len, _) = bincode::decode_from_slice(&end[20..], config)?;
            return Ok(Footer { opt_len });
        }
        Err(Error::new(
            "Footer does not start with the TRAC magic number".to_string(),
        ))
    }

    /// Calculate how much of the binary to trim off the end to not read the footer
    #[must_use]
    pub fn trim(&self) -> usize {
        // In order to not read the footer we need to trim 28 bytes plus any optional footer
        (28 + self.opt_len) as usize
    }
}
