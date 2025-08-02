# Cart-rs

Cart is a file format for storing and transfering malware in a safe way.

The primary way the file is made safe is by encrypting it with rc4 to prevent accidental
execution. Files are also zipped with zlib to minimize space usage.

Cart-rs does currently support any of the optional fields in the header or footer that the 
[Cart Spec](https://bitbucket.org/cse-assemblyline/cart/src/master/) allows. We do not plan
to ever support the optional footer fields as this would come at the cost of supporting streaming
downloads. This library is largely intended to allow the Thorium API to stream
CARTed files to and from s3.

# Examples

## CARTing a file

```
use tokio::fs::File;
use tokio::io::{BufReader, AsyncWriteExt};
use cart_rs::CartStream;
use generic_array::{typenum::U16, GenericArray};

// have a cart password to use
let password: GenericArray<u8, U16> =
   GenericArray::clone_from_slice(&"SecretCornIsBest".as_bytes()[..16]);
// build our cart streamer
let mut cart = CartStream::new(&password, 16384)?;
// Have a file to write our carted data too
let mut dest = File::create("/tmp/CartedFile").await?;
// wrap our open carted file handle in a BufReader
let mut reader = BufReader::new(source);
// read our carted file in chunks
while !cart.exhausted {
   // read some of our carted file to our buffer
   cart.reader(&mut reader).await?;
   // we have more data to cart so cart this part of the buffer
   let carted_bytes = cart.pack()?;
   // write our carted bytes
   dest.write_all(carted_bytes).await?;
}
 ```

 # UnCARTing a File

 ```
 use tokio::fs::{File, OpenOptions};
 use cart_rs::UncartStream;

// open our carted file
let carted = File::open("/tmp/CartedFile").await?;
// start uncarting this tream
let mut uncart = cart_rs::UncartStream::new(BufReader::new(carted));
// open a file to uncart too
let mut uncarted = OpenOptions::new()
   .read(true)
   .write(true)
   .create(true)
   .truncate(true)
   .open("/tmp/UnCartedFile")
   .await?;
// write our uncarted file to disk
tokio::io::copy(&mut uncart, &mut uncarted).await?;
 ```
