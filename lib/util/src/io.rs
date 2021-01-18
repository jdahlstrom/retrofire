use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::ops::DerefMut;

use crate::Buffer;
use crate::color::{Color, rgb};

pub fn load_ppm(filename: &str) -> Result<Buffer<Color>, Box<dyn Error>> {
    let mut r = BufReader::new(File::open(filename)?);

    let mut head = String::new();
    for _ in 0..4 {
        r.read_line(&mut head)?;
    }
    let mut head = head.split_whitespace();

    assert_eq!("P6", head.next().unwrap());
    let width: usize = head.next().unwrap().parse()?;
    let height: usize = head.next().unwrap().parse()?;
    assert_eq!(255, head.next().unwrap().parse::<usize>()?);
    assert_eq!(None, head.next());

    let mut data = Vec::with_capacity(3 * width * height);
    r.read_to_end(&mut data)?;

    assert_eq!(3 * width * height, data.len());

    let data = data
        .chunks(3)
        .map(|c| rgb(c[0], c[1], c[2]))
        .collect();

    Ok(Buffer { width, height, data })
}

pub fn save_ppm<B>(filename: &str, buf: &Buffer<Color, B>)
                   -> Result<(), Box<dyn Error>>
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