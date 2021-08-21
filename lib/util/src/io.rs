use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::ops::DerefMut;
use std::str::FromStr;

use crate::Buffer;
use crate::color::{BLACK, Color, gray, rgb, WHITE};

struct PnmHeader {
    magic: [u8; 2],
    width: usize,
    height: usize,
    #[allow(unused)]
    max: u16,
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

    loop {
        src.read_exact(c)?;
        if !c[0].is_ascii_whitespace() { break }
    }
    loop {
        res.push(c[0] as char);
        src.read_exact(c)?;
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
        if ![b"P4", b"P5", b"P6"].contains(&&magic) {
            return Err(pnm_parse_err("Invalid magic number"));
        }

        let width: usize = parse_ssv(src)?;
        let height: usize = parse_ssv(src)?;
        let max: u16 = if magic == *b"P4" { 1 } else { parse_ssv(src)? };

        Ok(PnmHeader { magic, width, height, max })
    }
}

pub fn load_pnm(filename: &str) -> Result<Buffer<Color>, Box<dyn Error>> {
    let r = &mut BufReader::new(File::open(filename)?);
    let h = PnmHeader::parse(r)?;

    let mut read_data = |len| {
        let mut data = Vec::with_capacity(len);
        r.read_to_end(&mut data)?;
        (data.len() == len).then(|| data)
            .ok_or_else(|| pnm_parse_err("Invalid data length"))
    };

    let data = match &h.magic {
        b"P6" => {
            read_data(3 * h.width * h.height)?
                .chunks(3)
                .map(|c| rgb(c[0], c[1], c[2]))
                .collect()
        }
        b"P5" => {
            read_data(h.width * h.height)?
                .into_iter()
                .map(gray)
                .collect()
        }
        b"P4" => {
            read_data(h.width * h.height / 8)?
                .into_iter()
                .flat_map(|c| (0..8).rev().map(move |i| c & (1 << i) != 0))
                .map(|c| if c { BLACK } else { WHITE })
                .collect()
        }
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
    writeln!(w, "P6 {} {} 255", buf.width, buf.height)?;

    let bytes: Vec<u8> = buf.data.iter()
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

        assert_eq!(*b"P6", hdr.magic);
        assert_eq!(123, hdr.width);
        assert_eq!(321, hdr.height);
        assert_eq!(255, hdr.max);
    }

    #[test]
    fn parse_header_p5() {
        let hdr = PnmHeader::parse(&mut &b"P5 123 456 64 "[..]).unwrap();

        assert_eq!(*b"P5", hdr.magic);
        assert_eq!(123, hdr.width);
        assert_eq!(456, hdr.height);
        assert_eq!(64, hdr.max);
    }

    #[test]
    fn parse_header_p4() {
        let hdr = PnmHeader::parse(&mut &b"P4 111 222 "[..]).unwrap();
        assert_eq!(*b"P4", hdr.magic);
        assert_eq!(111, hdr.width);
        assert_eq!(222, hdr.height);
        assert_eq!(1, hdr.max);
    }

    #[test]
    fn parse_header_invalid_magic() {
        let res = PnmHeader::parse(&mut &b"P1 1 1 1 "[..]);
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
