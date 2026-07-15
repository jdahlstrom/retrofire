//! PNM, also known as NetPBM, file format support.
//!
//! PNM is a venerable family of extremely simple image formats, each
//! consisting of a simple textual header followed by either textual or
//! binary pixel data.
//!
//! ```text
//! Type  | Txt | Bin | Pixel format
//! ------+-----+-----+-----------------
//! PBM   | P1  | P4  | 1 bpp monochrome
//! PGM   | P2  | P5  | 8 bpp grayscale
//! PPM   | P3  | P6  | 3x8 bpp RGB
//! ```

use alloc::{string::String, vec::Vec};
use core::{
    fmt::{self, Debug, Display, Formatter},
    iter::repeat_with,
    num::{IntErrorKind, ParseIntError, TryFromIntError},
};

use crate::math::{Color3, color::gray};

use super::{Buf2, Dims};

#[cfg(feature = "std")]
use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Bytes},
    path::Path,
};

#[cfg(feature = "std")]
use crate::render::Colorbuf;

#[cfg(feature = "std")]
use super::{
    AsSlice2, IntoPixel,
    pixfmt::{Rgb888, Xrgb8888},
};

use Error::*;
use Format::*;

/// Trait for writing PNM images.
///
/// This trait is implemented for various `Buf2` instances, so writing a buffer
/// is as easy as:
/// ```
/// use retrofire_core::util::{Buf2, Dims, pnm::WritePnm};
///
/// let buf = Buf2::new_from(Dims(3, 3), [0u8, 1, 2, 3, 4, 5, 6, 7, 8]);
///
/// buf.save("grays.pgm").unwrap();
/// ```
#[cfg(feature = "std")]
pub trait WritePnm {
    /// Writes `self` in PNM format to a writer.
    ///
    /// The exact format is specified by the implementation.
    fn write(&self, dest: impl io::Write) -> io::Result<()>;

    /*/// Encodes `self` in PNM format into a byte iterator.
    ///
    /// The exact format is specified by the implementation.
    fn encode(&self) -> Result<impl Iterator<Item = u8>> {
        todo!();
        Ok([].into_iter())
    }*/

    /// Writes `self` in PNM format to a file.
    ///
    /// The exact format is specified by the implementation.
    ///
    /// **Caution:** This function overwrites the file if it already exists.
    /// Use [`write`][Self::write] for more control over file creation.
    ///
    fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let dest = BufWriter::new(File::create(path)?);
        self.write(dest)
    }
}

/// Trait for reading and decoding PNM images.
///
/// Various `Buf2` instances implement this trait, so reading an image into
/// a buffer is as simple as:
/// ```should_panic
/// use retrofire_core::math::Color3;
/// use retrofire_core::util::{Buf2, pnm::ReadPnm};
///
/// let rgb_buf: Buf2<Color3> = Buf2::load("image.ppm").unwrap();
/// ```
/// or
/// ```should_panic
/// # use retrofire_core::util::{Buf2, pnm::ReadPnm};
/// let gray_buf: Buf2<u8> = Buf2::load("image.pgm").unwrap();
/// ```
///
/// If the `std` feature is not enabled, this trait only supports reading from
/// byte iterators. With the feature enabled, it is also possible to read from
/// `io::Read` or `Path` instances.
pub trait ReadPnm: Sized {
    /// Decodes a PNM-formatted image from a byte stream.
    fn decode(src: impl IntoIterator<Item = u8>) -> Result<Self>;

    /// Reads a PNM-formatted image from a reader.
    #[cfg(feature = "std")]
    fn read(src: impl io::Read) -> Result<Self> {
        struct Reader<R> {
            iter: Bytes<R>,
            res: Result<()>,
        }
        impl<R: io::Read> Iterator for Reader<R> {
            type Item = u8;

            fn next(&mut self) -> Option<Self::Item> {
                let res = self.iter.next()?;
                if let Err(e) = &res {
                    self.res = Err(Io(e.kind()))
                }
                res.ok()
            }
        }
        let mut reader = Reader { iter: src.bytes(), res: Ok(()) };
        let decode_res = Self::decode(reader.by_ref());
        reader.res.and(decode_res)
    }

    /// Reads a PNM-formatted image from a file path.
    #[cfg(feature = "std")]
    fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        Self::read(BufReader::new(file))
    }
}

/// The header of a PNM image.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Header {
    format: Format,
    dims: Dims,
    #[allow(unused)]
    // TODO Currently not used
    max: u16,
}

/// The format of a PNM image.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[allow(unused)]
#[repr(u16)]
pub enum Format {
    /// 1-bit monochrome image, text encoding.
    TextBitmap = magic(b"P1"),
    /// Grayscale image, text encoding.
    TextGraymap = magic(b"P2"),
    /// RGB image, text encoding.
    TextPixmap = magic(b"P3"),
    /// 1-bit monochrome image, packed binary encoding.
    BinaryBitmap = magic(b"P4"),
    /// Grayscale image, binary encoding. 1 byte per pixel.
    BinaryGraymap = magic(b"P5"),
    /// RGB image, binary encoding. 3 bytes per pixel.
    BinaryPixmap = magic(b"P6"),
}

// Error during loading or decoding a PNM file.
#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    /// An I/O error occurred.
    #[cfg(feature = "std")]
    Io(io::ErrorKind),
    /// Unsupported magic number.
    Unsupported([u8; 2]),
    /// Unexpected format.
    UnexpectedFormat(Format, u16),
    /// Unexpected end of input while decoding.
    UnexpectedEnd,
    /// Invalid numeric value encountered.
    InvalidNumber,
}

/// Result of loading or decoding a PNM file.
pub type Result<T> = core::result::Result<T, Error>;

//
// Inherent impls
//

impl Header {
    /// Attempts to parse a PNM header from an iterator.
    ///
    /// Currently supported formats are P2, P3, P4, P5, and P6.
    fn parse(input: impl IntoIterator<Item = u8>) -> Result<Self> {
        let mut it = input.into_iter();
        let magic = [
            it.next().ok_or(UnexpectedEnd)?,
            it.next().ok_or(UnexpectedEnd)?,
        ];
        let format = magic.try_into()?;
        let w = parse_u16(&mut it)?.into();
        let h = parse_u16(&mut it)?.into();
        let max: u16 = match &format {
            TextBitmap | BinaryBitmap => 1,
            _ => parse_u16(&mut it)?,
        };
        Ok(Self { format, dims: Dims(w, h), max })
    }
    /// Writes `self` to `dest` as a valid PNM header,
    /// including a trailing newline.
    #[cfg(feature = "std")]
    fn write(&self, mut dest: impl io::Write) -> io::Result<()> {
        let Self { format, dims: Dims(w, h), max } = *self;
        if matches!(format, TextBitmap | BinaryBitmap) {
            writeln!(dest, "{format} {w} {h}")
        } else {
            writeln!(dest, "{format} {w} {h} {max}")
        }
    }
}

impl Format {
    pub fn magic(&self) -> [u8; 2] {
        (*self as u16).to_be_bytes()
    }
}

//
// Local trait impls
//
#[cfg(feature = "std")]
impl<B> WritePnm for Colorbuf<B, Xrgb8888>
where
    B: AsSlice2<Elem = u32>,
{
    fn write(&self, mut dest: impl io::Write) -> io::Result<()> {
        let slice = self.buf.as_slice2();
        Header {
            format: BinaryPixmap,
            dims: slice.dims(),
            max: 255,
        }
        .write(&mut dest)?;

        slice.iter().try_for_each(|rgb| {
            let [_, rgb @ ..] = rgb.to_be_bytes();
            dest.write_all(&rgb)
        })
    }
}

#[cfg(feature = "std")]
impl<B> WritePnm for B
where
    B: AsSlice2,
    B::Elem: IntoPixel<[u8; 3], Rgb888> + Copy,
{
    /// Writes the contents of this buffer in binary PPM (P6) format.
    ///
    /// Each element of `self` is converted to an RGB triplet with `IntoPixel`.
    fn write(&self, mut dest: impl io::Write) -> io::Result<()> {
        let slice = self.as_slice2();
        Header {
            format: BinaryPixmap,
            dims: slice.dims(),
            max: 255,
        }
        .write(&mut dest)?;
        slice
            .iter()
            .map(|c| c.into_pixel(Rgb888))
            .try_for_each(|rgb| dest.write_all(&rgb[..]))
    }
}

#[cfg(feature = "std")]
impl WritePnm for Buf2<u8> {
    fn write(&self, mut dest: impl io::Write) -> io::Result<()> {
        Header {
            format: BinaryGraymap,
            dims: self.dims(),
            max: 255,
        }
        .write(&mut dest)?;
        dest.write_all(self.data())
    }
}

#[cfg(feature = "std")]
impl WritePnm for Buf2<bool> {
    fn write(&self, mut dest: impl io::Write) -> io::Result<()> {
        let slice = self.as_slice2();
        Header {
            format: BinaryBitmap,
            dims: slice.dims(),
            max: 1,
        }
        .write(&mut dest)?;

        let mut byte = 0;
        for (i, &bit) in (0..8).rev().cycle().zip(self.data()) {
            byte |= (bit as u8) << i;
            if i == 0 {
                dest.write_all(&[byte])?;
                byte = 0;
            }
        }
        if byte != 0 {
            // write remaining bits if size not divisible by 8
            dest.write_all(&[byte])?;
        }
        Ok(())
    }
}

impl ReadPnm for Buf2<Color3> {
    /// Decodes a PNM byte stream into a color buffer.
    ///
    /// # Supported Formats
    /// Currently supports the following PNM formats:
    /// * binary bitmap (P4): zero bits are mapped to white, one bits to black,
    ///     as per the PBM convention;
    /// * text and binary graymap (P2 and P5): each pixel value to the
    ///     corresponding gray;
    /// * text and binary pixmap (P3 and P6): each R,G,B triplet to the
    ///     corresponding RGB value.
    ///
    /// For all the non-bitmap formats, the maximum color or channel value
    /// in the header must be exactly 255. For more information on the PNM
    /// formats, see the [module documentation](self).
    fn decode(src: impl IntoIterator<Item = u8>) -> Result<Self> {
        decode_pnm(src)
    }
}

impl ReadPnm for Buf2<u8> {
    /// Decodes a PBM or PGM image from a stream of bytes.
    ///
    /// # Supported formats
    /// Currently supported PNM formats are:
    /// * Text graymap (P2) with maximum value exactly 255.
    /// * Binary graymap (P5) with maximum value exactly 255.
    /// * Binary bitmap (P4): 0-bits map to 255, 1-bits to 0.
    ///     In PBM conventionally zero means white and one means black.
    ///
    /// Read more about the formats in the [module docs][self].
    fn decode(src: impl IntoIterator<Item = u8>) -> Result<Self> {
        let mut it = src.into_iter();
        let h = Header::parse(it.by_ref())?;

        let count = h.dims.count();
        let data: Vec<_> = match h.format {
            TextGraymap if h.max == 255 => repeat_with(|| {
                let val = parse_u16(&mut it)?;
                Ok(u8::try_from(val)?)
            })
            .take(count)
            .collect::<Result<_>>()?,
            BinaryBitmap => decode_binary_bitmap(h.dims, it)
                .into_iter()
                // Conventionally in PBM 0 is white and 1 is black
                .map(|bit| 0xFF * (1 - bit as u8))
                .collect(),
            BinaryGraymap => it.take(count).collect(),
            fmt => return Err(UnexpectedFormat(fmt, h.max)),
        };
        if data.len() != count {
            return Err(UnexpectedEnd);
        }
        Ok(Self::new_from(h.dims, data))
    }
}

impl ReadPnm for Buf2<bool> {
    /// Decodes a PBM image from a stream of bytes.
    ///
    /// Currently only supports binary bitmaps (P4).
    /// Read more about the formats in the [module docs][self].
    fn decode(src: impl IntoIterator<Item = u8>) -> Result<Self> {
        let mut it = src.into_iter();
        let h = Header::parse(it.by_ref())?;

        if !matches!(h.format, BinaryBitmap) {
            return Err(UnexpectedFormat(h.format, h.max));
        }
        let data: Vec<_> = decode_binary_bitmap(h.dims, it);
        if data.len() != h.dims.count() {
            return Err(UnexpectedEnd);
        }
        Ok(Buf2::new_from(h.dims, data))
    }
}

/// Decodes a PNM image from a stream of bytes.
///
/// Currently supported PNM formats are P2, P3, P4, P5, and P6.
/// Read more about the formats in the [module docs][self].
///
/// # Errors
/// Returns [`Error`] in case of an invalid or unrecognized PNM image.
fn decode_pnm(input: impl IntoIterator<Item = u8>) -> Result<Buf2<Color3>> {
    let mut it = input.into_iter();
    let h = Header::parse(&mut it)?;

    let count = h.dims.count();
    let data: Vec<Color3> = match h.format {
        BinaryPixmap => decode_binary_pixmap(count, it),
        BinaryGraymap => it.map(gray).take(count).collect(),
        BinaryBitmap => decode_binary_bitmap(h.dims, it)
            .into_iter()
            // Conventionally in PBM 0 is white and 1 is black
            .map(|bit| gray(0xFF * (1 - bit as u8)))
            .collect(),
        TextPixmap => parse_text_pixmap(count, it)?,
        TextGraymap => repeat_with(|| {
            let val = parse_u16(&mut it)?;
            Ok(gray(val.try_into()?))
        })
        .take(count)
        .collect::<Result<Vec<_>>>()?,
        _ => return Err(Unsupported(h.format.magic())),
    };

    if data.len() < count {
        return Err(UnexpectedEnd);
    }
    Ok(Buf2::new_from(h.dims, data))
}

fn parse_text_pixmap(
    count: usize,
    mut it: impl Iterator<Item = u8>,
) -> Result<Vec<Color3>> {
    let mut col = [0u8; 3];
    (0..3)
        .cycle()
        .flat_map(|i| {
            let u16 = match parse_u16(&mut it) {
                Ok(u) => u,
                Err(e) => return Some(Err(e)),
            };
            // TODO Support >8-bit numbers
            col[i] = match u16.try_into() {
                Ok(u) => u,
                Err(e) => return Some(Err(e.into())),
            };
            (i == 2).then(|| Ok(col.into()))
        })
        .take(count)
        .collect()
}

fn decode_binary_bitmap(dims: Dims, it: impl Iterator<Item = u8>) -> Vec<bool> {
    // In P4, pixel values are packed in bytes, most significant bit first.
    // Each row is padded to the next byte boundary, taking ⌈width/8⌉ bytes.
    // For example, a 3x3 image takes three bytes, but a 9x1 image
    // only takes two.
    let mut data = Vec::new();
    let w_bits = dims.0 as usize;
    let w_bytes = w_bits.div_ceil(8) as usize;
    let n_bytes = w_bytes * dims.1 as usize;
    let mut it = it.take(n_bytes).peekable();
    while it.peek().is_some() {
        // For each row
        (&mut it)
            .take(w_bytes)
            .flat_map(|byte| (0..8).rev().map(move |i| (byte >> i) & 1))
            .take(w_bits)
            .for_each(|bit| {
                data.push(bit != 0);
            });
    }
    data
}

fn decode_binary_pixmap(
    count: usize,
    it: impl Iterator<Item = u8>,
) -> Vec<Color3> {
    let mut col = [0u8; 3];
    it.zip((0..3).cycle())
        .flat_map(|(c, i)| {
            col[i] = c;
            (i == 2).then(|| col.into())
        })
        .take(count)
        .collect()
}

/// Parses a numeric value from `src`, skipping whitespace and comments.
#[inline]
fn parse_u16(src: impl IntoIterator<Item = u8>) -> Result<u16> {
    fn do_parse(src: &mut dyn Iterator<Item = u8>) -> Result<u16> {
        let mut whitespace_or_comment = {
            let mut in_comment = false;
            move |b: &u8| match *b {
                b'#' => {
                    in_comment = true;
                    true
                }
                b'\n' => {
                    in_comment = false;
                    true
                }
                _ => in_comment || b.is_ascii_whitespace(),
            }
        };
        let str = src
            .into_iter()
            .skip_while(whitespace_or_comment)
            .take_while(|b| !whitespace_or_comment(b))
            .map(char::from)
            .collect::<String>();
        Ok(str.parse()?)
    }
    do_parse(&mut src.into_iter())
}

const fn magic(bytes: &[u8; 2]) -> u16 {
    u16::from_be_bytes(*bytes)
}

//
// Foreign trait impls
//

impl Display for Format {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use fmt::Write;
        let [p, n] = self.magic();
        f.write_char(p as char)?;
        f.write_char(n as char)
    }
}

impl TryFrom<[u8; 2]> for Format {
    type Error = Error;
    fn try_from(magic: [u8; 2]) -> Result<Self> {
        Ok(match &magic {
            b"P2" => TextGraymap,
            b"P3" => TextPixmap,
            b"P4" => BinaryBitmap,
            b"P5" => BinaryGraymap,
            b"P6" => BinaryPixmap,
            other => Err(Unsupported(*other))?,
        })
    }
}

impl core::error::Error for Error {}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            #[cfg(feature = "std")]
            Io(kind) => write!(f, "i/o error {kind}"),
            Unsupported([c, d]) => {
                use fmt::Write;
                f.write_str("unsupported magic number ")?;
                f.write_char(c as char)?;
                f.write_char(d as char)
            }
            UnexpectedFormat(fmt, max) => {
                write!(f, "unexpected format: {fmt}, max={max}")
            }
            UnexpectedEnd => f.write_str("unexpected end of input"),
            InvalidNumber => f.write_str("invalid numeric value"),
        }
    }
}

impl From<ParseIntError> for Error {
    fn from(e: ParseIntError) -> Self {
        match e.kind() {
            IntErrorKind::Empty => UnexpectedEnd,
            _ => InvalidNumber,
        }
    }
}
impl From<TryFromIntError> for Error {
    fn from(_: TryFromIntError) -> Self {
        // TODO better handling of unexpected values
        InvalidNumber
    }
}
#[cfg(feature = "std")]
impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Io(e.kind())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::math::rgb;

    #[test]
    fn parse_value_int() {
        assert_eq!(parse_u16(*b"123"), Ok(123));
        assert_eq!(parse_u16(*b"12345"), Ok(12345));
    }

    #[test]
    fn parse_num_empty() {
        assert_eq!(parse_u16(*b""), Err(UnexpectedEnd));
    }
    #[test]
    fn parse_num_too_large() {
        assert_eq!(parse_u16(*b"123456"), Err(InvalidNumber));
    }

    #[test]
    fn parse_num_with_whitespace() {
        assert_eq!(parse_u16(*b" \n\n   42 "), Ok(42));
    }

    #[test]
    fn parse_num_with_comment_before() {
        assert_eq!(parse_u16(*b"# this is a comment\n42"), Ok(42));
    }

    #[test]
    fn parse_num_with_comment_after() {
        assert_eq!(parse_u16(*b"42#this is a comment"), Ok(42));
    }

    #[test]
    fn parse_header_whitespace() {
        assert_eq!(
            Header::parse(*b"P6 123\t \n\r321      255 "),
            Ok(Header {
                format: BinaryPixmap,
                dims: Dims(123, 321),
                max: 255,
            })
        );
    }

    #[test]
    fn parse_header_comment() {
        assert_eq!(
            Header::parse(*b"P6 # foo 42\n 123\n#bar\n#baz\n321 255 "),
            Ok(Header {
                format: BinaryPixmap,
                dims: Dims(123, 321),
                max: 255,
            })
        );
    }

    #[test]
    fn parse_header_p2() {
        assert_eq!(
            Header::parse(*b"P2 123 456 789"),
            Ok(Header {
                format: TextGraymap,
                dims: Dims(123, 456),
                max: 789,
            })
        );
    }

    #[test]
    fn parse_header_p3() {
        assert_eq!(
            Header::parse(*b"P3 123 456 789"),
            Ok(Header {
                format: TextPixmap,
                dims: Dims(123, 456),
                max: 789,
            })
        );
    }

    #[test]
    fn parse_header_p4() {
        assert_eq!(
            Header::parse(*b"P4 123 456 "),
            Ok(Header {
                format: BinaryBitmap,
                dims: Dims(123, 456),
                max: 1,
            })
        );
    }

    #[test]
    fn parse_header_p5() {
        assert_eq!(
            Header::parse(*b"P5 123 456 789 "),
            Ok(Header {
                format: BinaryGraymap,
                dims: Dims(123, 456),
                max: 789,
            })
        );
    }

    #[test]
    fn parse_header_p6() {
        assert_eq!(
            Header::parse(*b"P6 123 456 789 "),
            Ok(Header {
                format: BinaryPixmap,
                dims: Dims(123, 456),
                max: 789,
            })
        );
    }

    #[test]
    fn parse_header_unsupported_magic() {
        let res = Header::parse(*b"P7 1 1 1 ");
        assert_eq!(res, Err(Unsupported(*b"P7")));
    }

    #[test]
    fn parse_header_invalid_magic() {
        let res = Header::parse(*b"FOO");
        assert_eq!(res, Err(Unsupported(*b"FO")));
    }

    #[test]
    fn parse_header_invalid_dims() {
        assert_eq!(Header::parse(*b"P5 abc 1 1 "), Err(InvalidNumber));
        assert_eq!(Header::parse(*b"P5 1 1 "), Err(UnexpectedEnd));
        assert_eq!(Header::parse(*b"P6 1 -1 1 "), Err(InvalidNumber));
    }

    #[test]
    fn parse_pnm_truncated() {
        let data = *b"P3 2 2 256 \n 0 0 0   123 0 42   0 64 128";
        assert_eq!(Buf2::<Color3>::decode(data), Err(UnexpectedEnd));
    }

    #[cfg(feature = "std")]
    #[test]
    fn write_header_p1() {
        let mut out = Vec::new();
        let hdr = Header {
            format: TextBitmap,
            dims: Dims(123, 456),
            max: 1,
        };
        hdr.write(&mut out).unwrap();
        assert_eq!(&out, b"P1 123 456\n");
    }

    #[cfg(feature = "std")]
    #[test]
    fn write_header_p6() {
        let mut out = Vec::new();
        let hdr = Header {
            format: BinaryPixmap,
            dims: Dims(123, 456),
            max: 789,
        };
        hdr.write(&mut out).unwrap();
        assert_eq!(&out, b"P6 123 456 789\n");
    }

    #[test]
    fn read_text_graymap_to_u8() {
        let data = *b"P2 2 2 255 \n 12 34 56 78";

        let buf: Buf2<u8> = Buf2::decode(data).unwrap();

        assert_eq!(buf.width(), 2);
        assert_eq!(buf.height(), 2);

        assert_eq!(buf[[0, 0]], 12);
        assert_eq!(buf[[1, 0]], 34);
        assert_eq!(buf[[0, 1]], 56);
        assert_eq!(buf[[1, 1]], 78);
    }

    #[test]
    fn read_text_graymap_invalid_max() {
        let data = *b"P2 2 2 128 \n 12 34 56 78";

        assert_eq!(
            Buf2::<u8>::decode(data),
            Err(UnexpectedFormat(TextGraymap, 128))
        );
    }

    #[test]
    fn read_text_graymap_to_color3() {
        let data = *b"P2 2 2 128 \n 12 34 56 78";

        let buf: Buf2<Color3> = Buf2::decode(data).unwrap();

        assert_eq!(buf.width(), 2);
        assert_eq!(buf.height(), 2);

        assert_eq!(buf[[0, 0]], rgb(12, 12, 12));
        assert_eq!(buf[[1, 0]], rgb(34, 34, 34));
        assert_eq!(buf[[0, 1]], rgb(56, 56, 56));
        assert_eq!(buf[[1, 1]], rgb(78, 78, 78));
    }

    #[test]
    fn read_text_pixmap_to_rgb() {
        let data = *b"P3 2 2 256 \n 0 0 0   123 0 42   0 64 128   255 255 255";

        let buf: Buf2<Color3> = Buf2::decode(data).unwrap();

        assert_eq!(buf.dims(), Dims(2, 2));

        assert_eq!(buf[[0, 0]], rgb(0, 0, 0));
        assert_eq!(buf[[1, 0]], rgb(123, 0, 42));
        assert_eq!(buf[[0, 1]], rgb(0, 64, 128));
        assert_eq!(buf[[1, 1]], rgb(255, 255, 255));
    }

    #[test]
    fn read_binary_bitmap_to_u8() {
        // In P4 each row is padded to the next byte boundary
        // -> 0110
        //    1001
        // = 0b0110_0000 0b1001_0000 = 0x60 0x90
        let buf: Buf2<u8> = Buf2::decode(*b"P4 4 2\n\x60\x90").unwrap();

        assert_eq!(buf.dims(), Dims(4, 2));

        assert_eq!(buf[0usize], [255, 0, 0, 255]);
        assert_eq!(buf[1usize], [0, 255, 255, 0]);
    }

    #[test]
    fn read_binary_bitmap_to_bool() {
        // 0x69 == 0b0110_1001
        let buf: Buf2<bool> = Buf2::decode(*b"P4 4 2\n\x69").unwrap();

        assert_eq!(buf.dims(), Dims(4, 2));

        assert_eq!(buf[0usize], [false, true, true, false]);
        assert_eq!(buf[1usize], [true, false, false, true]);
    }

    #[test]
    fn read_binary_graymap_to_color3() {
        let buf: Buf2<Color3> =
            Buf2::decode(*b"P5 2 2 255\n\x01\x23\x45\x67").unwrap();

        assert_eq!(buf.dims(), Dims(2, 2));

        assert_eq!(buf[0usize], [rgb(0x01, 0x01, 0x01), rgb(0x23, 0x23, 0x23)]);
        assert_eq!(buf[1usize], [rgb(0x45, 0x45, 0x45), rgb(0x67, 0x67, 0x67)]);
    }

    #[test]
    fn read_binary_pixmap_to_color3() {
        let buf: Buf2<Color3> = Buf2::decode(
            *b"P6 2 2 255\n\
            \x01\x12\x23\
            \x34\x45\x56\
            \x67\x78\x89\
            \x9A\xAB\xBC",
        )
        .unwrap();

        assert_eq!(buf.dims(), Dims(2, 2));

        assert_eq!(buf[0usize], [rgb(0x01, 0x12, 0x23), rgb(0x34, 0x45, 0x56)]);
        assert_eq!(buf[1usize], [rgb(0x67, 0x78, 0x89), rgb(0x9A, 0xAB, 0xBC)]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn write_bool_to_binary_bitmap() {
        let buf = Buf2::new_from(
            Dims(4, 2),
            [false, true, true, false, true, false, false, true],
        );

        let mut out = Vec::new();
        buf.write(&mut out).unwrap();

        assert_eq!(out, b"P4 4 2\n\x69");
    }

    #[cfg(feature = "std")]
    #[test]
    fn write_color3_to_binary_pixmap() {
        use alloc::vec;
        let buf = Buf2::new_from(
            Dims(2, 2),
            vec![
                rgb(0xFF, 0, 0),
                rgb(0, 0xFF, 0),
                rgb(0, 0, 0xFF),
                rgb(0xFF, 0xFF, 0),
            ],
        );

        let mut out = Vec::new();
        buf.write(&mut out).unwrap();

        assert_eq!(
            &out,
            b"P6 2 2 255\n\
              \xFF\x00\x00\
              \x00\xFF\x00\
              \x00\x00\xFF\
              \xFF\xFF\x00"
        );
    }
}
