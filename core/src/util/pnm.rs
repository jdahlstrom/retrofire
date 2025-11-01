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
    num::{IntErrorKind, ParseIntError},
    str::FromStr,
};
#[cfg(feature = "std")]
use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Read, Write},
    path::Path,
};

use crate::math::{Color3, rgb};

use super::{Dims, buf::Buf2};
#[cfg(feature = "std")]
use super::{
    buf::AsSlice2,
    pixfmt::{IntoPixel, Rgb888},
};

use crate::math::color::gray;
use Error::*;
use Format::*;

/// The header of a PNM image.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Header {
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
enum Format {
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

const fn magic(bytes: &[u8; 2]) -> u16 {
    u16::from_be_bytes(*bytes)
}

impl Display for Format {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "P{}", *self as u8 as char)
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

// Error during loading or decoding a PNM file.
#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    /// An I/O error occurred.
    #[cfg(feature = "std")]
    Io(io::ErrorKind),
    /// Unsupported magic number.
    Unsupported([u8; 2]),
    /// Unexpected magic number.
    Unexpected(Format),
    /// Unexpected end of input while decoding.
    UnexpectedEnd,
    /// Invalid numeric value encountered.
    InvalidNumber,
}

/// Result of loading or decoding a PNM file.
pub type Result<T> = core::result::Result<T, Error>;

impl core::error::Error for Error {}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            #[cfg(feature = "std")]
            Io(kind) => write!(f, "i/o error {kind}"),
            Unsupported([c, d]) => {
                write!(f, "unsupported magic number {}{}", c as char, d as char)
            }
            Unexpected(fmt) => {
                write!(f, "unexpected format {fmt}")
            }
            UnexpectedEnd => f.write_str("unexpected end of input"),
            InvalidNumber => f.write_str("invalid numeric value"),
        }
    }
}

impl From<ParseIntError> for Error {
    fn from(e: ParseIntError) -> Self {
        if *e.kind() == IntErrorKind::Empty {
            UnexpectedEnd
        } else {
            InvalidNumber
        }
    }
}

#[cfg(feature = "std")]
impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Io(e.kind())
    }
}

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
        let dims = (parse_num(&mut it)?, parse_num(&mut it)?);
        let max: u16 = match &format {
            TextBitmap | BinaryBitmap => 1,
            _ => parse_num(&mut it)?,
        };
        Ok(Self { format, dims, max })
    }
    /// Writes `self` to `dest` as a valid PNM header,
    /// including a trailing newline.
    #[cfg(feature = "std")]
    fn write(&self, mut dest: impl Write) -> io::Result<()> {
        let Self { format, dims: (w, h), max } = *self;
        let max: &dyn Display = match format {
            TextBitmap | BinaryBitmap => &"",
            _ => &max,
        };
        writeln!(dest, "{format} {w} {h} {max}")
    }
}

/// Loads a PNM image from a path into a buffer.
///
/// Currently supported formats are P2, P3, P4, P5, and P6.
/// Read more about the formats in the [module docs][self].
///
/// # Errors
/// Returns [`pnm::Error`][Error] in case of an I/O error or invalid PNM image.
#[cfg(feature = "std")]
pub fn load_pnm(path: impl AsRef<Path>) -> Result<Buf2<Color3>> {
    let r = &mut BufReader::new(File::open(path)?);
    read_pnm(r)
}

/// Reads a PNM image into a buffer.
///
/// Currently supported PNM formats are P2, P3, P4, P5, and P6.
/// Read more about the formats in the [module docs][self].
///
/// # Errors
/// Returns [`pnm::Error`][Error] in case of an I/O error or invalid PNM image.
#[cfg(feature = "std")]
pub fn read_pnm(input: impl Read) -> Result<Buf2<Color3>> {
    let input = BufReader::new(input);
    parse_pnm(input.bytes().map_while(io::Result::ok))
}

fn parse_binary_bitmap(
    it: &mut impl Iterator<Item = u8>,
) -> impl Iterator<Item = bool> {
    it.flat_map(|byte| {
        (0..8) //
            .rev()
            .map(move |i| ((byte >> i) & 1) != 0)
    })
}
fn parse_binary_pixmap(
    it: &mut impl Iterator<Item = u8>,
) -> impl Iterator<Item = [u8; 3]> {
    let mut col = [0u8; 3];
    it.zip((0..3).cycle()).flat_map(move |(c, i)| {
        col[i] = c;
        (i == 2).then(|| col.into())
    })
}
fn parse_text_pixmap<'a>(
    it: &'a mut impl Iterator<Item = u8>,
) -> impl Iterator<Item = Result<[u8; 3]>> + 'a {
    let mut col = [0u8; 3];
    //let it = &it;
    (0..3).cycle().flat_map(move |i| {
        col[i] = match parse_num(it) {
            Ok(c) => c,
            Err(e) => return Some(Err(e)),
        };
        (i == 2).then(|| Ok(col))
    })
}

/// Attempts to decode a PNM image from an iterator of bytes.
///
/// Currently supported PNM formats are P2, P3, P4, P5, and P6.
/// Read more about the formats in the [module docs][self].
///
/// # Errors
/// Returns [`Error`] in case of an invalid or unrecognized PNM image.
pub fn parse_pnm(input: impl IntoIterator<Item = u8>) -> Result<Buf2<Color3>> {
    let mut it = input.into_iter();
    let h = Header::parse(&mut it)?;

    let count = h.dims.0 * h.dims.1;
    let data: Vec<Color3> = match h.format {
        BinaryPixmap => parse_binary_pixmap(&mut it)
            .take(count as usize)
            .map(Into::into)
            .collect(),
        BinaryGraymap => it //
            .map(|c| rgb(c, c, c))
            .take(count as usize)
            .collect(),
        BinaryBitmap => parse_binary_bitmap(&mut it)
            // Conventionally in PBM 0 is white, 1 is black
            .map(|bit| gray(if bit { 0x00 } else { 0xFF }))
            .take(count as usize)
            .collect(),
        TextPixmap => parse_text_pixmap(&mut it)
            .map(|res| res.map(Into::into))
            .take(count as usize)
            .collect::<Result<Vec<_>>>()?,
        TextGraymap => (0..count)
            .map(|_| {
                let val = parse_num(&mut it)?;
                Ok(rgb(val, val, val))
            })
            .take(count as usize)
            .collect::<Result<Vec<_>>>()?,
        // TODO just support TextBitmap too, even though it's pretty useless
        _ => unreachable!("caught by the header parser"),
    };

    if data.len() < count as usize {
        Err(UnexpectedEnd)
    } else {
        Ok(Buf2::new_from(h.dims, data))
    }
}

pub fn parse_pbm_or_pgm(
    input: impl IntoIterator<Item = u8>,
) -> Result<Buf2<u8>> {
    let mut it = input.into_iter();
    let h = Header::parse(&mut it)?;
    let data = match h.format {
        BinaryGraymap => it.collect(),
        BinaryBitmap => parse_binary_bitmap(&mut it)
            // Conventionally in PBM 0 is white, 1 is black
            .map(|bit| if bit { 0x00 } else { 0xFF })
            .collect(),
        TextGraymap => (0..h.dims.0 * h.dims.1)
            .map(|_| parse_num(&mut it))
            .collect::<Result<Vec<_>>>()?,
        f => return Err(Unexpected(f)),
    };
    Ok(Buf2::new_from(h.dims, data))
}

/// Writes an image to a file in PPM format, P6 sub-format
/// (binary 8-bits-per-channel RGB).
///
/// Caution: This function overwrites the file if it already exists.
/// Use [`write_ppm`] for more control over file creation.
///
/// # Errors
/// Returns [`std::io::Error`] if an error occurs while writing.
#[cfg(feature = "std")]
pub fn save_ppm<T>(
    path: impl AsRef<Path>,
    data: impl AsSlice2<T>,
) -> io::Result<()>
where
    T: IntoPixel<[u8; 3], Rgb888> + Copy,
{
    let out = BufWriter::new(File::create(path)?);
    write_ppm(out, data)
}

/// Writes an image to `out` in PPM format, P6 sub-format
/// (binary 8-bits-per-channel RGB).
///
/// # Errors
/// Returns [`std::io::Error`] if an error occurs while writing.
#[cfg(feature = "std")]
pub fn write_ppm<T>(
    mut out: impl Write,
    data: impl AsSlice2<T>,
) -> io::Result<()>
where
    T: IntoPixel<[u8; 3], Rgb888> + Copy,
{
    let slice = data.as_slice2();
    Header {
        format: BinaryPixmap,
        dims: slice.dims(),
        max: 255,
    }
    .write(&mut out)?;

    // Appease the borrow checker
    slice
        .rows()
        .flatten()
        .map(|c| c.into_pixel())
        .try_for_each(|rgb| out.write_all(&rgb[..]))
}

/// Parses a numeric value from `src`, skipping whitespace and comments.
fn parse_num<T>(src: &mut impl Iterator<Item = u8>) -> Result<T>
where
    T: FromStr,
    Error: From<T::Err>,
{
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
        .skip_while(whitespace_or_comment)
        .take_while(|b| !whitespace_or_comment(b))
        .map(char::from)
        .collect::<String>();
    Ok(str.parse()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_value_int() {
        assert_eq!(parse_num(*b"123"), Ok(123));
        assert_eq!(parse_num(*b"12345"), Ok(12345));
    }

    #[test]
    fn parse_num_empty() {
        assert_eq!(parse_num::<i32>(*b""), Err(UnexpectedEnd));
    }

    #[test]
    fn parse_num_with_whitespace() {
        assert_eq!(parse_num(*b" \n\n   42 "), Ok(42));
    }

    #[test]
    fn parse_num_with_comment_before() {
        assert_eq!(parse_num(*b"# this is a comment\n42"), Ok(42));
    }

    #[test]
    fn parse_num_with_comment_after() {
        assert_eq!(parse_num(*b"42#this is a comment"), Ok(42));
    }

    #[test]
    fn parse_header_whitespace() {
        assert_eq!(
            Header::parse(*b"P6 123\t \n\r321      255 "),
            Ok(Header {
                format: BinaryPixmap,
                dims: (123, 321),
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
                dims: (123, 321),
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
                dims: (123, 456),
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
                dims: (123, 456),
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
                dims: (123, 456),
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
                dims: (123, 456),
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
                dims: (123, 456),
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
        assert_eq!(parse_pnm(data).err(), Some(UnexpectedEnd));
    }

    #[cfg(feature = "std")]
    #[test]
    fn write_header_p1() {
        let mut out = Vec::new();
        let hdr = Header {
            format: TextBitmap,
            dims: (123, 456),
            max: 1,
        };
        hdr.write(&mut out).unwrap();
        assert_eq!(&out, b"P1 123 456 \n");
    }

    #[cfg(feature = "std")]
    #[test]
    fn write_header_p6() {
        let mut out = Vec::new();
        let hdr = Header {
            format: BinaryPixmap,
            dims: (123, 456),
            max: 789,
        };
        hdr.write(&mut out).unwrap();
        assert_eq!(&out, b"P6 123 456 789\n");
    }

    #[test]
    fn read_pnm_p2() {
        let data = *b"P2 2 2 128 \n 12 34 56 78";

        let buf = parse_pnm(data).unwrap();

        assert_eq!(buf.width(), 2);
        assert_eq!(buf.height(), 2);

        assert_eq!(buf[[0, 0]], rgb(12, 12, 12));
        assert_eq!(buf[[1, 0]], rgb(34, 34, 34));
        assert_eq!(buf[[0, 1]], rgb(56, 56, 56));
        assert_eq!(buf[[1, 1]], rgb(78, 78, 78));
    }

    #[test]
    fn read_pnm_p3() {
        let data = *b"P3 2 2 256 \n 0 0 0   123 0 42   0 64 128   255 255 255";

        let buf = parse_pnm(data).unwrap();

        assert_eq!(buf.dims(), (2, 2));

        assert_eq!(buf[[0, 0]], rgb(0, 0, 0));
        assert_eq!(buf[[1, 0]], rgb(123, 0, 42));
        assert_eq!(buf[[0, 1]], rgb(0, 64, 128));
        assert_eq!(buf[[1, 1]], rgb(255, 255, 255));
    }

    #[test]
    fn read_pnm_p4() {
        // 0x69 == 0b0110_1001
        let buf = parse_pnm(*b"P4 4 2\n\x69").unwrap();

        assert_eq!(buf.dims(), (4, 2));

        let b = rgb(0u8, 0, 0);
        let w = rgb(0xFFu8, 0xFF, 0xFF);

        assert_eq!(buf[0usize], [w, b, b, w]);
        assert_eq!(buf[1usize], [b, w, w, b]);
    }

    #[test]
    fn read_pnm_p5() {
        let buf = parse_pnm(*b"P5 2 2 255\n\x01\x23\x45\x67").unwrap();

        assert_eq!(buf.dims(), (2, 2));

        assert_eq!(buf[0usize], [rgb(0x01, 0x01, 0x01), rgb(0x23, 0x23, 0x23)]);
        assert_eq!(buf[1usize], [rgb(0x45, 0x45, 0x45), rgb(0x67, 0x67, 0x67)]);
    }

    #[test]
    fn read_pnm_p6() {
        let buf = parse_pnm(
            *b"P6 2 2 255\n\
            \x01\x12\x23\
            \x34\x45\x56\
            \x67\x78\x89\
            \x9A\xAB\xBC",
        )
        .unwrap();

        assert_eq!(buf.dims(), (2, 2));

        assert_eq!(buf[0usize], [rgb(0x01, 0x12, 0x23), rgb(0x34, 0x45, 0x56)]);
        assert_eq!(buf[1usize], [rgb(0x67, 0x78, 0x89), rgb(0x9A, 0xAB, 0xBC)]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn write_ppm() {
        use alloc::vec;
        let buf = vec![
            rgb(0xFF, 0, 0),
            rgb(0, 0xFF, 0),
            rgb(0, 0, 0xFF),
            rgb(0xFF, 0xFF, 0),
        ];

        let mut out = vec![];
        super::write_ppm(&mut out, Buf2::new_from((2, 2), buf)).unwrap();

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
