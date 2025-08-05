//! Cart is a file format for storing and transferring malware in a safe way.
//!
//! The primary way the file is made safe is by encrypting it with rc4 to prevent accidental
//! execution. Files are also zipped with zlib to minimize space usage.
//!
//! Cart-rs does not currently support any of the optional fields in the header or footer that the
//! [Cart Spec](https://bitbucket.org/cse-assemblyline/cart/src/master/) allows. We do not plan
//! to ever support the optional footer fields as this would come at the cost of supporting streaming
//! downloads. This library is largely intended to allow the Thorium API to stream
//! `CaRTed` files to and from s3.
//!
//! # Examples
//!
//! ## `CaRTing` a File
//!
//! ```
//! use tokio::fs::{File, OpenOptions};
//! use tokio::io::{BufReader, AsyncWriteExt};
//! use cart_rs::CartStream;
//! use generic_array::{typenum::U16, GenericArray};
//!
//! # async fn exec() -> Result<(), cart_rs::Error> {
//! # let mut source = File::create("/tmp/EvilCorn").await?;
//! # source.write_all("ImMalware".as_bytes()).await?;
//! # // open our file in read mode
//! # let source = File::open("/tmp/EvilCorn").await?;
//! // have a cart password to use
//! let password: GenericArray<u8, U16> =
//!     GenericArray::clone_from_slice(&"SecretCornIsBest".as_bytes()[..16]);
//! // build our cart stream
//! let mut cart_stream = CartStream::new(BufReader::new(source), &password)?;
//! // open a file to cart to
//! let mut output = OpenOptions::new()
//!     .read(true)
//!     .write(true)
//!     .create(true)
//!     .truncate(true)
//!     .open("/tmp/CartedCorn")
//!     .await?;
//! // write our carted file to disk
//! tokio::io::copy(&mut cart_stream, &mut output).await?;
//! # // validate our file was carted correctly
//! # // open our carted file
//! # let carted = File::open("/tmp/CartedCorn").await?;
//! # // start uncarting this stream
//! # let mut uncart_stream = cart_rs::UncartStream::new(BufReader::new(carted));
//! # // open a file to uncart to
//! # let mut uncarted = tokio::fs::OpenOptions::new()
//! #     .read(true)
//! #     .write(true)
//! #     .create(true)
//! #     .truncate(true)
//! #     .open("/tmp/UnCartedCorn")
//! #     .await?;
//! # // write our uncarted file to disk
//! # tokio::io::copy(&mut uncart_stream, &mut uncarted).await?;
//! # // make sure the files match
//! # let uncarted = tokio::fs::read_to_string("/tmp/UnCartedCorn").await?;
//! # if uncarted != "ImMalware" {
//! #      panic!("{} is not 'ImMalware'", uncarted);
//! # }
//! # std::fs::remove_file("/tmp/EvilCorn")?;
//! # std::fs::remove_file("/tmp/CartedCorn")?;
//! # std::fs::remove_file("/tmp/UnCartedCorn")?;
//! # Ok(())
//! # }
//!
//! # tokio_test::block_on(async {
//! #    exec().await.unwrap()
//! # })
//! ```
//!
//! ## `UnCaRTing` a File
//!
//! ```
//! use tokio::fs::{File, OpenOptions};
//! use cart_rs::UncartStream;
//! # use tokio::io::{BufReader, AsyncWriteExt, AsyncReadExt};
//! # use cart_rs::CartStreamManual;
//! # use generic_array::{typenum::U16, GenericArray};
//! use bytes::BytesMut;
//!
//! # async fn exec() -> Result<(), cart_rs::Error> {
//! # let mut source = File::create("/tmp/MaliciousPotato").await?;
//! # source.write_all("ImMalware".as_bytes()).await?;
//! # // open our file in read mode
//! # let source = File::open("/tmp/MaliciousPotato").await?;
//! # // have a cart password to use
//! # let password: GenericArray<u8, U16> =
//! #     GenericArray::clone_from_slice(&"SecretCornIsBest".as_bytes()[..16]);
//! # // build our cart streamer
//! # let mut cart = CartStreamManual::new(&password, 16384)?;
//! # // Have a file to write our carted data to
//! # let mut dest = File::create("/tmp/CartedPotato").await?;
//! # // wrap our open carted file handle in a BufReader
//! # let mut reader = BufReader::new(source);
//! # // Allocate a bytesmut to write too
//! # let mut temp = BytesMut::zeroed(32_768);
//! # // read our carted file in chunks
//! # loop {
//! #   // read in some bytes
//! #   let bytes_read = reader.read(&mut temp).await.unwrap();
//! #   // check how many bytes were read
//! #   if bytes_read == 0 {
//! #       break;
//! #   }
//! #   // freeze the section of bytes we have written data too
//! #   let frozen = temp.split_to(bytes_read).freeze();
//! #   // add these bytes to our cart stream
//! #   if cart.next_bytes(frozen).unwrap() {
//! #     // keep processing these bytes until they are finished
//! #     while cart.process().unwrap() {
//! #       // if we have more than 5 MiB worth of data then write to disk
//! #       if cart.ready() >= 5_242_800 {
//! #           // get the bytes we have ready to write
//! #           let writable = cart.carted_bytes();
//! #           // write our packed bytes
//! #           dest.write_all(writable).await.unwrap();
//! #           // consume these written bytes
//! #           cart.consume();
//! #       }
//! #     }
//! #   }
//! # }
//! # // finish packing our cart file
//! # let buff = cart.finish().unwrap();
//! # // write our final bytes
//! # dest.write_all(buff).await.unwrap();
//!  // open our carted file
//!  let carted = File::open("/tmp/CartedPotato").await?;
//!  // start uncarting this tream
//!  let mut uncart = cart_rs::UncartStream::new(BufReader::new(carted));
//!  // open a file to uncart to
//!  let mut uncarted = OpenOptions::new()
//!     .read(true)
//!     .write(true)
//!     .create(true)
//!     .truncate(true)
//!     .open("/tmp/UnCartedPotato")
//!     .await?;
//!  // write our uncarted file to disk
//!  tokio::io::copy(&mut uncart, &mut uncarted).await?;
//! # // make sure the files match
//! # let uncarted = tokio::fs::read_to_string("/tmp/UnCartedPotato").await?;
//! # if uncarted != "ImMalware" {
//! #      panic!("{} is not 'ImMalware'", uncarted);
//! # }
//! # std::fs::remove_file("/tmp/MaliciousPotato")?;
//! # std::fs::remove_file("/tmp/CartedPotato")?;
//! # std::fs::remove_file("/tmp/UnCartedPotato")?;
//! # Ok(())
//! # }
//!
//! # tokio_test::block_on(async {
//! #    exec().await.unwrap()
//! # })
//!```
//!
//! ## `CaRTing` a file manually
//!
//! ```
//! use tokio::fs::File;
//! use tokio::io::{BufReader, AsyncWriteExt, AsyncReadExt};
//! use cart_rs::CartStreamManual;
//! use generic_array::{typenum::U16, GenericArray};
//! use bytes::BytesMut;
//!
//! # async fn exec() -> Result<(), cart_rs::Error> {
//! # let mut source = File::create("/tmp/WickedSquash").await?;
//! # source.write_all("ImMalware".as_bytes()).await?;
//! # // open our file in read mode
//! # let source = File::open("/tmp/WickedSquash").await?;
//! // have a cart password to use
//! let password: GenericArray<u8, U16> =
//!     GenericArray::clone_from_slice(&"SecretCornIsBest".as_bytes()[..16]);
//! // build our cart streamer
//! let mut cart = CartStreamManual::new(&password, 16384)?;
//! // Have a file to write our carted data to
//! let mut dest = File::create("/tmp/CartedSquash").await?;
//! // wrap our open carted file handle in a BufReader
//! let mut reader = BufReader::new(source);
//! // Allocate a bytesmut to write too
//! let mut temp = BytesMut::zeroed(32_768);
//! // read our carted file in chunks
//! loop {
//!   // read in some bytes
//!   let bytes_read = reader.read(&mut temp).await.unwrap();
//!   // check how many bytes were read
//!   if bytes_read == 0 {
//!       break;
//!   }
//!   // freeze the section of bytes we have written data too
//!   let frozen = temp.split_to(bytes_read).freeze();
//!   // add these bytes to our cart stream
//!   if cart.next_bytes(frozen).unwrap() {
//!     // keep processing these bytes until they are finished
//!     while cart.process().unwrap() {
//!       // if we have more than 5 MiB worth of data then write to disk
//!       if cart.ready() >= 5_242_800 {
//!           // get the bytes we have ready to write
//!           let writable = cart.carted_bytes();
//!           // write our packed bytes
//!           dest.write_all(writable).await.unwrap();
//!           // consume these written bytes
//!           cart.consume();
//!       }
//!     }
//!   }
//! }
//! // finish packing our cart file
//! let buff = cart.finish().unwrap();
//! // write our final bytes
//! dest.write_all(buff).await.unwrap();
//! # // validate our file was carted correctly
//! # // open our carted file
//! # let carted = File::open("/tmp/CartedSquash").await?;
//! # // start uncarting this tream
//! # let mut uncart = cart_rs::UncartStream::new(BufReader::new(carted));
//! # // open a file to uncart to
//! # let mut uncarted = tokio::fs::OpenOptions::new()
//! #    .read(true).write(true).create(true).truncate(true)
//! #    .open("/tmp/UnCartedSquash").await?;
//! # // write our uncarted file to disk
//! # tokio::io::copy(&mut uncart, &mut uncarted).await?;
//! # // make sure the files match
//! # let uncarted = tokio::fs::read_to_string("/tmp/UnCartedSquash").await?;
//! # if uncarted != "ImMalware" {
//! #      panic!("{} is not 'ImMalware'", uncarted);
//! # }
//! # std::fs::remove_file("/tmp/WickedSquash")?;
//! # std::fs::remove_file("/tmp/CartedSquash")?;
//! # std::fs::remove_file("/tmp/UnCartedSquash")?;
//! # Ok(())
//! # }
//!
//! # tokio_test::block_on(async {
//! #    exec().await.unwrap()
//! # })
//! ```
use bytes::{Buf, Bytes};
use crypto::rc4::Rc4;
use crypto::symmetriccipher::SynchronousStreamCipher;
use flate2::{Compress, Compression, FlushCompress, Status};
use futures_core::ready;
use generic_array::{ArrayLength, GenericArray};
use miniz_oxide::inflate::stream::InflateState;
use miniz_oxide::MZFlush;
use rc4::{KeyInit, StreamCipher};
use std::convert::TryFrom;
use std::io::prelude::*;
use std::io::ErrorKind;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::io::{AsyncBufRead, AsyncRead, ReadBuf};

mod errors;
mod libs;

pub use errors::Error;
pub use libs::{footer, footer::Footer, header, header::Header};

/// Packs a Cart file using streaming
#[pin_project::pin_project]
pub struct CartStream<R: AsyncBufRead, T: ArrayLength<u8>> {
    /// The input file stream to cart
    #[pin]
    pub input: R,
    /// The key used for encryption
    key: GenericArray<u8, T>,
    /// The zlib compressor to use
    zlib: Compress,
    /// The rc4 encryptor to encrypt our data with
    rc4: rc4::Rc4<T>,
    /// Signals that the header has been written
    header_written: bool,
    /// Signals that the file has been completely carted and the footer has been appended
    footer_written: bool,
    /// Signals that the input has been exhausted and all data has been written to the output
    finished: bool,
}

impl<R: AsyncBufRead, T: ArrayLength<u8>> CartStream<R, T> {
    /// Create a new cart stream to cart a file
    ///
    /// # Arguments
    ///
    /// * `input` - A reader for the file to cart
    /// * `key` - The 16 byte key to use for encryption
    pub fn new(input: R, key: &GenericArray<u8, T>) -> Result<Self, Error> {
        // check that the given key is valid
        Header::validate_key(key)?;
        // build the buffer to store data ready to be compressed, decrypted, and written
        // build our compressor
        let zlib = Compress::new(Compression::default(), true);
        // build our rc4 encryptor
        let rc4 = rc4::Rc4::new(key);
        let cart = CartStream {
            input,
            key: key.clone(),
            zlib,
            rc4,
            header_written: false,
            footer_written: false,
            finished: false,
        };
        Ok(cart)
    }

    /// Write the `CaRT` header to the internal read buffer, returning the number of bytes written
    ///
    /// # Arguments
    ///
    /// * `buf` - The internal buffer to write carted data to
    fn write_header(self: Pin<&mut Self>, buf: &mut ReadBuf<'_>) -> Result<usize, std::io::Error> {
        // write the header if it hasn't been written already
        let this = self.project();
        let output = buf.initialize_unfilled();
        if let Err(err) = Header::write(this.key, output) {
            return Err(std::io::Error::new(ErrorKind::InvalidData, err));
        }
        *this.header_written = true;
        Ok(header::HEADER_LEN)
    }

    /// Write the `CaRT` footer to the internal read buffer, returning the number of bytes written
    ///
    /// # Arguments
    ///
    /// * `buf` - The internal buffer to write carted data to
    fn write_footer(self: Pin<&mut Self>, buf: &mut ReadBuf<'_>) -> Result<usize, std::io::Error> {
        let this = self.project();
        let output = buf.initialize_unfilled();
        if let Err(err) = Footer::write(output) {
            return Err(std::io::Error::new(ErrorKind::InvalidData, err));
        }
        // signal the footer has been written and the file is complete
        *this.footer_written = true;
        Ok(footer::FOOTER_LEN)
    }

    /// Try to read from our stream and cart any data to the internal read buffer
    ///
    /// # Arguments
    ///
    /// * `cx` - The current context
    /// * `buf` - The internal buffer to write carted data to
    fn do_poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        // project our struct
        let mut this = self.project();
        // get the total number of bytes output by compressor before compressing this round
        let old_total_out = this.zlib.total_out();
        // get a reference to the next unfilled chunk of the internal read buffer
        let output = buf.initialize_unfilled();
        // ingest and compress data until the compressor flushes data to the output buffer
        'compress: loop {
            // read in the next chunk of bytes
            let chunk = ready!(this.input.as_mut().poll_fill_buf(cx))?;
            let flush = if chunk.is_empty() {
                // if the input is exhausted, signal the compressor to just
                // flush data from its internal buffer
                FlushCompress::Finish
            } else {
                FlushCompress::None
            };
            let old_total_in = this.zlib.total_in();
            // compress the data and write it to the output buffer
            // Note: when using "FlushCompress::None", the compressor decides when to flush data from
            //       its internal buffer to the output buffer, meaning this call could 1) only consume data
            //       from the input for compression on subsequent calls, 2) only flush compressed data to
            //       the output without consuming any data from the input, or 3) both consume input and
            //       flush compressed data to the output; all 3 possibilities are accounted for
            match this.zlib.compress(chunk, output, flush) {
                Ok(status) => match status {
                    Status::Ok => (),
                    Status::BufError => {
                        return Poll::Ready(Err(std::io::Error::new(
                            ErrorKind::Other,
                            "Zip Compression Buffer Error",
                        )));
                    }
                    Status::StreamEnd => {
                        // signal that the carting process has finished and exit the loop
                        *this.finished = true;
                        break 'compress;
                    }
                },
                Err(err) => return Poll::Ready(Err(std::io::Error::new(ErrorKind::Other, err))),
            };
            // calculate the number of bytes consumed in compression
            let bytes_in = (this.zlib.total_in() - old_total_in) as usize;
            // mark bytes as consumed from the input
            this.input.as_mut().consume(bytes_in);
            if this.zlib.total_out() != old_total_out {
                // exit the loop if any compressed data was written to the output buffer
                break 'compress;
            }
        }
        // calculate the number of bytes that were output by the compressor
        let compressed_out = (this.zlib.total_out() - old_total_out) as usize;
        // encrypt the compressed data
        this.rc4.apply_keystream(&mut output[..compressed_out]);
        // advance the internal buffer by the number of bytes written
        buf.advance(compressed_out);
        Poll::Ready(Ok(()))
    }
}

impl<R: AsyncBufRead, T: ArrayLength<u8>> AsyncRead for CartStream<R, T> {
    /// Perform a poll for data, compressing, encrypting, and writing
    /// any available data to the internal read buffer
    ///
    /// # Arguments
    ///
    /// * `cx` - The current context
    /// * `buf` - The internal read buffer storing output carted data
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        if !self.header_written {
            // write the header if it hasn't been written
            match self.write_header(buf) {
                Ok(bytes_written) => {
                    // advance the buffer by the number of bytes written
                    buf.advance(bytes_written);
                    Poll::Ready(Ok(()))
                }
                Err(err) => Poll::Ready(Err(err)),
            }
        } else if !self.footer_written && self.finished {
            // write the footer if it hasn't been written and our input is exhausted
            match self.write_footer(buf) {
                Ok(bytes_written) => {
                    // advance the buffer by the number of bytes written
                    buf.advance(bytes_written);
                    Poll::Ready(Ok(()))
                }
                Err(err) => Poll::Ready(Err(err)),
            }
        } else if self.footer_written {
            // return immediately (signalling completion) if the footer was already written
            Poll::Ready(Ok(()))
        } else {
            // if the header has been written and we have more input,
            // attempt to read, compress, and encrypt more data
            self.do_poll_read(cx, buf)
        }
    }
}

/// Unpacks a Cart file using streaming
///
/// This Cart file cannot have any comments for this to work.
#[pin_project::pin_project]
pub struct UncartStream<R: AsyncBufRead> {
    /// The carted stream to read from
    #[pin]
    cart: R,
    /// The buffer to store decrypted but still compressed data
    decrypted: Vec<u8>,
    /// Where to start reading data to decompress from the decrypted buffer
    decrypt_start: usize,
    /// Where to end reading data to decompress from the decrypted buffer
    decrypt_end: usize,
    /// The buffer to store decompressed, finished data ready to be written to the read buffer
    decompressed: Vec<u8>,
    /// The rc4 decryptor to use for this file
    rc4: Option<Rc4>,
    /// The amount of decompressed bytes from the decompressed buffer yet to be written
    /// to the read buffer
    decompressed_remaining: usize,
    /// The number of decompressed bytes from the decompressed buffer already written
    /// to the read buffer
    decompressed_consumed: usize,
    /// The zlib decompressor to use
    zlib: Box<InflateState>,
}

impl<R: AsyncBufRead> UncartStream<R> {
    /// The size of the internal buffer storing decrypted data
    const DECRYPTED_BUF_SIZE: usize = 65536;
    /// The size of the internal buffer storing decompressed data
    const DECOMPRESSED_BUF_SIZE: usize = 131_072;

    /// Create a new uncart stream to uncart a file with no comments
    ///
    /// Using this to uncart a file with comments will likely fail and should not be done.
    ///
    /// # Arguments
    ///
    /// * `cart` - A reader for the file to uncart
    pub fn new(cart: R) -> Self {
        // build buffers to store decrypted+compressed and decompressed data
        let decrypted = vec![0; Self::DECRYPTED_BUF_SIZE];
        let decompressed = vec![0; Self::DECOMPRESSED_BUF_SIZE];
        UncartStream {
            cart,
            decrypted,
            decrypt_start: 0,
            decrypt_end: 0,
            decompressed,
            rc4: None,
            decompressed_remaining: 0,
            decompressed_consumed: 0,
            zlib: InflateState::new_boxed_with_window_bits(15),
        }
    }

    /// Try to read from our stream and uncart any data to the internal read buffer
    ///
    /// # Arguments
    ///
    /// * `cx` - The current context
    /// * `buf` - The internal read buffer to read uncarted data to
    fn do_poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        // get a reference to our output buffer
        let output = buf.initialize_unfilled();
        // pin our struct
        let mut this = self.project();
        // track the number of bytes we have returned this time
        let mut returned = 0;
        // copy decompressed remaining/consumed bytes counts locally so we are reentrant safe
        let mut local_remaining = *this.decompressed_remaining;
        let mut local_consumed = *this.decompressed_consumed;
        // track whether failing to get more data would cause us to lose data
        let mut write_hole = false;
        // keep reading until we have uncarted all input data or have filled our output buffer
        'decrypt_and_decompress: loop {
            // determine if we have more data to decompress still
            if *this.decompressed_remaining == 0 || local_remaining == 0 {
                // if we don't already have decrypted data to decompress then get more
                if this.decrypt_start == this.decrypt_end {
                    // determine if this is our first read by checking if the decryptor has been built
                    let first_read = this.rc4.is_none();
                    // read in and decrypt the next chunk from the input
                    let bytes_read = ready!(Self::read_and_decrypt(
                        &mut this.cart,
                        this.decrypted,
                        this.rc4,
                        cx
                    ))?;
                    if bytes_read == 0 {
                        // if no more data was read, return immediately, signalling that uncarting is complete
                        return Poll::Ready(Ok(()));
                    }
                    // mark bytes from the input CaRT as consumed
                    this.cart.as_mut().consume(bytes_read);
                    // set the end point to read from the decrypted buffer
                    *this.decrypt_end = if first_read {
                        // if this was the first read, subtract the length of the CaRT header
                        bytes_read - header::HEADER_LEN
                    } else {
                        bytes_read
                    };
                    // reset the decompression start position in the decrypted buffer
                    *this.decrypt_start = 0;
                    // reset the counter of decompressed bytes consumed
                    *this.decompressed_consumed = 0;
                }
                // select the decrypted slice to decompress
                let decrypted = &this.decrypted[*this.decrypt_start..*this.decrypt_end];
                // decompress our decrypted data
                let decompress_result = miniz_oxide::inflate::stream::inflate(
                    this.zlib,
                    decrypted,
                    this.decompressed,
                    MZFlush::None,
                );
                // increment the decompress start point by the number of bytes consumed from the decrypted buffer
                *this.decrypt_start += decompress_result.bytes_consumed;
                // return an error if one occurred in decompressing
                if let Err(err) = decompress_result.status {
                    match err {
                        miniz_oxide::MZError::Data | miniz_oxide::MZError::Buf => {
                            return Poll::Ready(Err(std::io::Error::new(
                                ErrorKind::InvalidData,
                                "CaRT file cannot be decompressed because data is missing/corrupted",
                            )));
                        }
                        _ => {
                            return Poll::Ready(Err(std::io::Error::new(
                                ErrorKind::InvalidData,
                                "An unknown error occurred while decompressing the carted data",
                            )))
                        }
                    }
                };
                // cap the end point to copy from the decompress buffer to the space left in the output buffer
                let decompress_end =
                    std::cmp::min(output.len() - returned, decompress_result.bytes_written);
                // get the slice of decompressed data to read into our output buffer
                let decompressed = &this.decompressed[..decompress_end];
                // get the slice of data to write new output data to
                let target = &mut output[returned..returned + decompress_end];
                // copy our readable data to our output slice
                target.copy_from_slice(decompressed);
                // update the number of bytes we have ready to return
                local_remaining = decompress_result.bytes_written - decompress_end;
                // update the number of bytes we have consumed
                local_consumed = decompressed.len();
                returned += decompress_end;
                // if we have filled our output buffer or polling for data could cause data loss then return
                if returned == output.len()
                    || write_hole
                    || decompress_result.bytes_written < output.len()
                {
                    break 'decrypt_and_decompress;
                }
            } else {
                // we still have decompressed data remaining so just write that;
                // cap the end point to copy from the decompress buffer to the space left in the output buffer
                let decompress_end = std::cmp::min(output.len() - returned, local_remaining);
                // get the slice of decompressed data to read into our output buffer
                let decompressed =
                    &this.decompressed[local_consumed..local_consumed + decompress_end];
                // get the slice of data to write new output data to
                let target = &mut output[returned..returned + decompress_end];
                // copy our readable data to our output slice
                target.copy_from_slice(decompressed);
                // update the amount of decompressed bytes we have remaining
                local_remaining -= decompressed.len();
                // track the local number of already decompressed bytes we have consumed
                local_consumed += decompressed.len();
                // update the number of bytes we have ready to return
                returned += target.len();
                // if we have filled our output buffer then return
                if returned == output.len() {
                    break 'decrypt_and_decompress;
                }
                // set our write hole since further polling could cause data loss
                write_hole = true;
            }
        }
        // persist remaining/consumed decompressed bytes count for next poll
        *this.decompressed_remaining = local_remaining;
        *this.decompressed_consumed = local_consumed;
        // advance the output buffer
        buf.advance(returned);
        Poll::Ready(Ok(()))
    }

    /// Read and decrypt data from the input buffer and store the result in the internal
    /// compressed buffer; returns the offset at which to start decompressing in the buffer the number of bytes
    /// that were read, the offset at which to start decompressing from the
    ///
    /// # Arguments
    ///
    /// * `cart` - The cart reader to read from
    /// * `decrypted` - The buffer to store decrypted data in
    /// * `rc4` - The RC4 decryptor that may or may not have been created yet
    fn read_and_decrypt(
        cart: &mut Pin<&mut R>,
        decrypted: &mut [u8],
        rc4: &mut Option<Rc4>,
        cx: &mut Context<'_>,
    ) -> Poll<std::io::Result<usize>> {
        // read in data from the input CaRT file
        let raw = ready!(cart.as_mut().poll_fill_buf(cx))?;
        if raw.is_empty() {
            if rc4.is_none() {
                // if this is our first read (the decryptor hasn't yet been built),
                // the file is empty so return an error
                return Poll::Ready(Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    "Input file is empty!",
                )));
            }
            // if this isn't our first read, the input is completely exhausted, so return
            return Poll::Ready(Ok(0));
        }
        // get our rc4 decryptor or create it if it doesn't exist
        let (rc4, decrypt_input, consumed) = match rc4 {
            // we have already built an rc4 decryptor so just use that
            Some(rc4) => {
                let decompressable = std::cmp::min(raw.len(), Self::DECRYPTED_BUF_SIZE);
                (rc4, &raw[..decompressable], decompressable)
            }
            None => {
                // build a decryptor from the decryption key contained in the first chunk
                // if one hasn't been created and store it for subsequent calls
                match Self::build_decryptor(raw) {
                    Ok(rc4_built) => {
                        let _ = rc4.insert(rc4_built);
                    }
                    Err(err) => {
                        return Poll::Ready(Err(err));
                    }
                }
                // update the number of bytes that are decompressable (subtracting the size of the header)
                let decompressable =
                    std::cmp::min(raw.len() - header::HEADER_LEN, Self::DECRYPTED_BUF_SIZE);
                // get a new slice of data that doesn't contain our header and is sized to our target buff
                let decrypt_input = &raw[header::HEADER_LEN..decompressable + header::HEADER_LEN];
                // return our rc4 encryptor and the bytes to decrypt
                (
                    rc4.as_mut().unwrap(),
                    decrypt_input,
                    decompressable + header::HEADER_LEN,
                )
            }
        };
        // select a slice the exact length of the input for the decrypt output
        let decrypt_output = &mut decrypted[..decrypt_input.len()];
        // decrypt this chunk of input data
        rc4.process(decrypt_input, decrypt_output);
        // return the number of bytes that were read in
        Poll::Ready(Ok(consumed))
    }

    fn build_decryptor(first_chunk: &[u8]) -> Result<Rc4, std::io::Error> {
        // if we haven't gotten enough bytes for the header yet then just skip to the next loop without consuming
        if first_chunk.len() < header::HEADER_LEN {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "Invalid CaRT file! CaRT header is malformed or missing.",
            ));
        }
        // try to read in our header
        let header = match Header::get(&first_chunk[..header::HEADER_LEN]) {
            Ok(header) => header,
            Err(err) => {
                return Err(std::io::Error::new(ErrorKind::InvalidData, err));
            }
        };
        // build our rc4 decryptor
        Ok(Rc4::new(&header.key))
    }
}

impl<R: AsyncBufRead> AsyncRead for UncartStream<R> {
    /// Poll to see if there is any uncarted data available
    ///
    /// # Arguments
    ///
    /// * `cx` - The current context
    /// * `buf` - The internal buffer to store available uncarted data
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        // check if our read buf is empty
        if buf.remaining() == 0 {
            return Poll::Ready(Ok(()));
        }
        // try to read more data
        self.do_poll_read(cx, buf)
    }
}

/// Packs files using the Cart format manually
///
/// This allows users to cart files on streams of data that do not implement
/// `AsyncRead` and instead are passed in as a stream of `Bytes`.
pub struct CartStreamManual<T: ArrayLength<u8>> {
    /// The index to start newly carted data at
    skip: usize,
    /// The next buffer to process after our current one is consumed
    on_deck: Option<Bytes>,
    /// The current buffer to procress
    current: Option<Bytes>,
    /// The zlib compressor to use
    zlib: Compress,
    /// The rc4 encryptor to encrypt our data with
    rc4: rc4::Rc4<T>,
    /// The vector to store our compressed and encrypted data
    output: Vec<u8>,
}

impl<T: ArrayLength<u8>> CartStreamManual<T> {
    /// Create a new manual cart stream buffer
    ///
    /// On the first write to the buffer start writing after the header.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to use when encrypting this data
    /// * `len` - The size of the buffer to allocate
    pub fn new(key: &GenericArray<u8, T>, len: usize) -> Result<Self, Error> {
        // pre allocate a new output buffer
        let output = Header::new_buffer(key, len + 768_432)?;
        //build our compressor
        let zlib = Compress::new(Compression::default(), true);
        // build our rc4 encryptor
        let rc4 = rc4::Rc4::new(key);
        let cart = CartStreamManual {
            // the first pack round will start after the header
            skip: header::HEADER_LEN,
            on_deck: None,
            current: None,
            zlib,
            rc4,
            output,
        };
        Ok(cart)
    }

    /// process some bytes
    ///
    /// Returns true if we have more data to cart from this buffer.
    ///
    /// # Arguments
    ///
    /// * `flush` - The flush setting to use
    fn cart_bytes(&mut self, flush: FlushCompress) -> Result<bool, Error> {
        // get our current buffer
        let Some(buff) = self.current.as_mut() else {
            return Ok(false);
        };
        // if our output buffer is full then just tell the user we have more data to cart
        if self.skip == self.output.len() {
            return Ok(true);
        }
        // get the old number of bytes that went in and out of our compressor
        let old_total = self.zlib.total_out();
        let old_in = self.zlib.total_in();
        // compress this input block
        let status = self
            .zlib
            .compress(&buff[..], &mut self.output[self.skip..], flush)
            .unwrap();
        // return an error if a failed status was returned
        if status == Status::BufError {
            println!("skip -> {}", self.skip);
            println!("outp -> {}", self.output.len());
            return Err(Error::new("Zip Compression Buffer Error".to_owned()));
        }
        // get the total number of bytes that were compressed in this loop
        let zipped = usize::try_from(self.zlib.total_out() - old_total)?;
        // get the index to stop encrypting at
        let zip_end = self.skip + zipped;
        // encrypt our compressed data
        self.rc4
            .apply_keystream(&mut self.output[self.skip..zip_end]);
        // update our skip value
        self.skip += zipped;
        // determine how many bytes were consumed from our input
        let consumed = self.zlib.total_in() - old_in;
        // advance our buffer
        buff.advance(consumed as usize);
        Ok(buff.has_remaining())
    }

    /// Add the next buffer to cart and start processing our new current buffer
    ///
    /// If you are using the reader method above then you do not need to call this.
    ///
    /// # Arguments
    ///
    /// * `raw` - The raw bytes to add
    pub fn next_bytes(&mut self, raw: Bytes) -> Result<bool, Error> {
        // always keep one buffer in hold to ensure we can do the final write correctly
        if let Some(old) = self.on_deck.replace(raw) {
            // set this buffer as our current buffer
            self.current = Some(old);
            // process our old bytes
            self.cart_bytes(FlushCompress::Partial)
        } else {
            Ok(false)
        }
    }

    /// Process the next chunk of bytes in a current buffer
    pub fn process(&mut self) -> Result<bool, Error> {
        // process more bytes in our current buffer
        self.cart_bytes(FlushCompress::Partial)
    }

    /// Get the number of carted bytes that are ready to be read
    pub fn ready(&self) -> usize {
        self.skip
    }

    /// Get a slice to the currently carted bytes
    pub fn carted_bytes(&self) -> &[u8] {
        &self.output[..self.skip]
    }

    /// Consume our currently carted bytes
    pub fn consume(&mut self) {
        self.skip = 0;
    }

    /// Finish packing this file and write the CART footer
    pub fn finish(&mut self) -> Result<&[u8], Error> {
        // get our last buffer
        let Some(buff) = self.on_deck.take() else {
            return Err(Error::FinishBeforeData);
        };
        // update our current buffer
        self.current = Some(buff);
        // process our old bytes
        self.cart_bytes(FlushCompress::Finish)?;
        // determine if we need to extend our output buffer
        if self.skip + footer::FOOTER_LEN > self.output.capacity() {
            // extend our buffer by the length of the footer
            self.output.try_reserve_exact(footer::FOOTER_LEN)?;
            // initialize our output buffer to zeros
            self.output.extend((0..footer::FOOTER_LEN).map(|_| 0));
        }
        // get a reference to the correct position to write our footer at
        let mut footer_buf = &mut self.output[self.skip..self.skip + footer::FOOTER_LEN];
        // zero our footer out
        unsafe {
            std::ptr::write_bytes(footer_buf.as_mut_ptr(), 0, footer_buf.len());
        }
        // write the final 4 bytes of our footer
        footer_buf.write_all(footer::MAGIC_NUM)?;
        Ok(&self.output[..self.skip + footer::FOOTER_LEN])
    }
}
