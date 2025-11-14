//! The headers for the cart format

use crate::Error;
use std::io::Write;

/// The length of a standard `CaRT` header
pub const HEADER_LEN: usize = 38;
/// The cart magic number preceding the header
pub static MAGIC_NUM: &[u8; 4] = b"CART";
/// The length of the key used for encryption
pub const KEY_LEN: usize = 16;

/// The mandatory header object for cart
///
/// The optional length will largely be ignored as we do not currently support optional headers.
#[derive(Debug, Clone)]
pub struct Header {
    /// The version of ``CaRT`` in use
    pub version: u8,
    /// The key used to encrypt this file
    pub key: Vec<u8>,
    /// The length of the optional header
    pub opt_len: usize,
}

impl Header {
    /// Build the header for this carted file
    ///
    /// # Arguments
    ///
    /// * `key` - The key used by rc4 for encryption
    ///
    /// # Errors
    ///
    /// If the given key is invalid (not exactly 16 bytes long), an error will be returned.
    pub fn new_buffer(key: &[u8], len: usize) -> Result<Vec<u8>, Error> {
        Self::validate_key(key)?;
        // build our header vector of 38 + the length requested
        let mut header: Vec<u8> = vec![0; HEADER_LEN + len];
        // write the header
        Header::write(key, &mut header[..HEADER_LEN])?;
        Ok(header)
    }

    /// Write this header to the start of an already allocated vec
    ///
    /// # Arguments
    ///
    /// * `key` - The key used by rc4 for encryption
    /// * `buff` - The buffer to write our header info to
    ///
    /// # Errors
    ///
    /// If the given key is invalid (not exactly 16 bytes long), an error will be returned.
    ///
    /// Additionally, if any IO errors occur then an error will be returned, and the header
    /// will fail to write. IO errors should only occur if an insufficient buffer is provided.
    pub fn write(key: &[u8], mut buf: &mut [u8]) -> Result<(), Error> {
        Self::validate_key(key)?;
        // create the bincode config; use fixed int encoding to ensure we write 8 bytes
        // when we write `0_u64` instead of just 1 byte
        let config = bincode::config::standard().with_fixed_int_encoding();
        // write the CaRT magic number (4 bytes)
        buf.write_all(MAGIC_NUM)?;
        // write CaRT version 1 (2 bytes)
        let version = b"\x01\x00";
        buf.write_all(version)?;
        // write reserved space (8 bytes)
        bincode::encode_into_std_write(0_u64, &mut buf, config)?;
        // write the encryption key (16 bytes)
        buf.write_all(key)?;
        // hardcode an optional header length of 0 (8 bytes)
        bincode::encode_into_std_write(0_u64, &mut buf, config)?;
        Ok(())
    }

    /// Gets the header from the first 38 bytes of the raw binary
    ///
    /// # Arguments
    ///
    /// * `raw` - The first 38 bytes of the binary containing the header
    ///
    /// # Errors
    ///
    /// If this buffer does not start with the CART magic number then an error will be returned.
    pub fn get(raw: &[u8]) -> Result<Self, Error> {
        // create the bincode config; use fixed int encoding because that's how we write things
        let config = bincode::config::standard().with_fixed_int_encoding();
        // make sure the magic numbers match carts magic number
        Self::validate(raw)?;
        // extract the version number
        let (version, _) = bincode::decode_from_slice(&raw[4..5], config)?;
        // extract the rc4 key
        let key = raw[14..30].to_vec();
        // extract the length of the optional header
        let (opt_len, _) = bincode::decode_from_slice(&raw[30..], config)?;
        Ok(Header {
            version,
            key,
            opt_len,
        })
    }

    pub fn validate(raw: &[u8]) -> Result<(), Error> {
        if raw.len() < 4 {
            return Err(Error::new(
                "Cannot validate Cart file because the given header buffer is empty or too small",
            ));
        } else if raw[..4] != *MAGIC_NUM {
            return Err(Error::new("File does not start with the CART magic number"));
        }
        Ok(())
    }

    /// Checks that the key is valid given CART specifications. The key must be exactly 16
    /// bytes log to be valid.
    ///
    /// # Arguments
    ///
    /// * `key` - The key used for encryption
    ///
    /// # Errors
    ///
    /// If the given key is invalid (not exactly 16 bytes long), an error will be returned.
    pub fn validate_key(key: &[u8]) -> Result<(), Error> {
        if key.len() != KEY_LEN {
            return Err(Error::new(format!(
                "The given key does not have the correct length of {}. Given key length: {}",
                KEY_LEN,
                key.len()
            )));
        }
        Ok(())
    }

    /// Calculate how much of the binary to skip to get past the header
    #[must_use]
    pub fn skip(&self) -> usize {
        // In order to skip the header we have to skip 38 bytes + the optional header size
        HEADER_LEN + self.opt_len
    }
}
