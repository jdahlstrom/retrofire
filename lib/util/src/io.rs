use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::ops::DerefMut;
use std::str::FromStr;

use crate::buf::Buffer;
use crate::color::{BLACK, Color, gray, rgb, WHITE};

#[derive(Copy, Clone, Debug)]
struct PnmHeader {
    magic: Magic,
    width: usize,
    height: usize,
    #[allow(unused)]
    max: u16,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Magic([char; 2]);

const BITMAP_TXT: Magic = Magic(['P', '1']);
#[allow(unused)]
const GRAYMAP_TXT: Magic = Magic(['P', '2']);
const PIXMAP_TXT: Magic = Magic(['P', '3']);
const BITMAP_BIN: Magic = Magic(['P', '4']);
const GRAYMAP_BIN: Magic = Magic(['P', '5']);
const PIXMAP_BIN: Magic = Magic(['P', '6']);

impl TryFrom<[u8; 2]> for Magic {
    type Error = Box<dyn Error>;
    fn try_from(magic: [u8; 2]) -> Result<Self, Self::Error> {
        if [b"P3", b"P4", b"P5", b"P6"].contains(&&magic) {
            Ok(Self(magic.map(char::from)))
        } else {
            Err(pnm_parse_err(format!("Unsupported magic number {magic:?}")))
        }
    }
}

#[derive(Debug)]
struct ParseError(String);

impl Display for ParseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error parsing PNM image: {}", self.0)
    }
}

impl Error for ParseError {}

fn parse_ssv<T>(src: &mut dyn Read) -> Result<T, Box<dyn Error>>
where
    T: FromStr,
    T::Err: ToString
{
    let mut res = String::new();
    let c = &mut [0];

    src.read_exact(c)?;
    loop {
        if !c[0].is_ascii_whitespace() { break }
        src.read_exact(c)?;
    }
    if c[0] == b'#' {
        loop {
            src.read_exact(c)?;
            if c[0] == b'\n' || c[0] == b'\r' { break }
        }
    }
    loop {
        if !c[0].is_ascii_whitespace() { break }
        src.read_exact(c)?;
    }
    loop {
        res.push(c[0] as char);
        if src.read(c)? == 0 { break }
        if c[0].is_ascii_whitespace() { break }
    }
    res.parse().map_err(|e| pnm_parse_err(e))
}

fn pnm_parse_err(reason: impl ToString) -> Box<dyn Error> {
    Box::new(ParseError(reason.to_string()))
}

impl PnmHeader {
    fn parse(src: &mut dyn Read) -> Result<PnmHeader, Box<dyn Error>> {
        let mut magic = [0u8; 2];
        src.read_exact(&mut magic)?;
        let magic = Magic::try_from(magic)?;
        let width: usize = parse_ssv(src)?;
        let height: usize = parse_ssv(src)?;
        let max: u16 = if matches!(magic, BITMAP_TXT | BITMAP_BIN) {
            1
        } else {
            parse_ssv(src)?
        };
        Ok(PnmHeader { magic, width, height, max })
    }
}

pub fn load_pnm(filename: &str) -> Result<Buffer<Color>, Box<dyn Error>> {
    let r = &mut BufReader::new(File::open(filename)?);
    read_pnm(r)
}

pub fn read_pnm(mut rd: impl Read) -> Result<Buffer<Color>, Box<dyn Error>> {

    let h = PnmHeader::parse(&mut rd)?;

    let mut read_data = |len| {
        let mut data = Vec::with_capacity(len);
        rd.read_to_end(&mut data)?;
        if data.len() == len {
            Ok(data)
        } else {
            Err(pnm_parse_err(format!(
                "Invalid data length {}, expected {}", data.len(), len)))
        }
    };

    let data = match h.magic {
        PIXMAP_BIN => {
            read_data(3 * h.width * h.height)?
                .chunks(3)
                .map(|c| rgb(c[0], c[1], c[2]))
                .collect()
        }
        GRAYMAP_BIN => {
            read_data(h.width * h.height)?
                .into_iter()
                .map(gray)
                .collect()
        }
        BITMAP_BIN => {
            read_data(h.width * h.height / 8)?
                .into_iter()
                .flat_map(|c| (0..8).rev().map(move |i| c & (1 << i) != 0))
                .map(|c| if c { BLACK } else { WHITE })
                .collect()
        }
        PIXMAP_TXT => {
            let mut data = Vec::with_capacity(2 * h.width * h.height);
            for _ in 0..h.width * h.height {
                let r = parse_ssv(&mut rd)?;
                let g = parse_ssv(&mut rd)?;
                let b = parse_ssv(&mut rd)?;
                data.push(rgb(r, g, b));
            }
            data
        }
        BITMAP_TXT => unimplemented!(),
        GRAYMAP_TXT => unimplemented!(),
        _ => unreachable!()
    };

    Ok(Buffer::from_vec(h.width, data))
}

pub fn save_ppm<B>(
    filename: &str,
    buf: &Buffer<Color, B>,
) -> Result<(), Box<dyn Error>>
where
    B: DerefMut<Target=[Color]>,
{
    let mut w = BufWriter::new(File::create(filename)?);
    writeln!(w, "P6 {} {} 255", buf.width(), buf.height())?;

    let bytes: Vec<u8> = buf.data().iter()
        .flat_map(|c| c.to_argb()[1..].to_vec())
        .collect();

    w.write_all(&bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_header_whitespace() {
        let mut hdr: &[u8] = b"P6 123\t \n\r321                      255 ";
        let hdr = PnmHeader::parse(&mut hdr).unwrap();

        assert_eq!(PIXMAP_BIN, hdr.magic);
        assert_eq!(123, hdr.width);
        assert_eq!(321, hdr.height);
        assert_eq!(255, hdr.max);
    }

    #[test]
    fn parse_header_comment() {
        let mut hdr: &[u8] = b"P6 # foo 42\n 123\n#bar\n321 255 ";
        let hdr = PnmHeader::parse(&mut hdr).unwrap();

        assert_eq!(PIXMAP_BIN, hdr.magic);
        assert_eq!(123, hdr.width);
        assert_eq!(321, hdr.height);
        assert_eq!(255, hdr.max);
    }

    #[test]
    fn parse_header_p5() {
        let hdr = PnmHeader::parse(&mut &b"P5 123 456 64 "[..]).unwrap();

        assert_eq!(GRAYMAP_BIN, hdr.magic);
        assert_eq!(123, hdr.width);
        assert_eq!(456, hdr.height);
        assert_eq!(64, hdr.max);
    }

    #[test]
    fn parse_header_p4() {
        let hdr = PnmHeader::parse(&mut &b"P4 111 222 "[..]).unwrap();
        assert_eq!(BITMAP_BIN, hdr.magic);
        assert_eq!(111, hdr.width);
        assert_eq!(222, hdr.height);
        assert_eq!(1, hdr.max);
    }

    #[test]
    fn parse_header_and_data_p3() {
        let data = b"P3 2 2 256 \n 0 0 0   123 0 42   0 64 128   255 255 255";

        let buf = read_pnm(&mut &data[..]).unwrap();

        assert_eq!(2, buf.width());
        assert_eq!(2, buf.height());
        assert_eq!(BLACK, *buf.get(0, 0));
        assert_eq!(rgb(123, 0, 42), *buf.get(1, 0));
        assert_eq!(rgb(0, 64, 128), *buf.get(0, 1));
        assert_eq!(WHITE, *buf.get(1, 1));
    }

    #[test]
    fn parse_header_unsupported_magic() {
        let res = PnmHeader::parse(&mut &b"P2 1 1 1 "[..]);
        assert!(res.is_err());
    }

    #[test]
    fn parse_header_invalid_magic() {
        let res = PnmHeader::parse(&mut &b"FOO"[..]);
        assert!(res.is_err());
    }

    #[test]
    fn parse_header_invalid_dims() {
        let res = PnmHeader::parse(&mut &b"P5 abc 1 1 "[..]);
        assert!(res.is_err());
        let res = PnmHeader::parse(&mut &b"P5 1 1 "[..]);
        assert!(res.is_err());
        let res = PnmHeader::parse(&mut &b"P6 1 -1 1 "[..]);
        assert!(res.is_err());
    }
}
