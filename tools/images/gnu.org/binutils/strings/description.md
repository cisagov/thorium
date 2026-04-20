Run the linux/binutils `strings` command to identify sequences of printable characters. There are several instantiations of this tool that look for different character encodings:

 * `strings`: single-7-bit-byte characters (default)
 * `strings-16be`: 16-bit bigendian
 * `strings-16le`: 16-bit littleendian
 * `strings-32be`: 32-bit bigendian
 * `strings-32le`: 32-bit littleendian