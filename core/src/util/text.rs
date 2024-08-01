use alloc::{string::ToString, vec::Vec};
use core::fmt::{Display, Formatter};

use super::buf::{AsMutSlice2, Buf2, Slice2};

pub struct Font<T> {
    glyphs: Atlas<T>,
    glyph_w: u32,
    glyph_h: u32,
}

#[allow(unused)]
pub struct Options {
    control_chars: bool,
    align: Align,
    char_spacing: i8,
    line_spacing: i8,
}

pub enum Align {
    Left,
    Center,
    Right,
}

impl<T> Font<T> {
    pub fn new(glyphs: Buf2<T>, glyph_w: u32, glyph_h: u32) -> Self {
        Self { glyphs, glyph_w, glyph_h }
    }

    /// Returns the glyph corresponding to `c`.
    pub fn glyph(&self, c: u8) -> Slice2<T> {
        let (col, row) = (c as u32 % 16, c as u32 / 16);
        let (gw, gh) = self.glyph_size();
        let (x0, y0) = (col * gw, row * gh);
        self.glyphs.slice((x0..x0 + gw, y0..y0 + gh))
    }

    pub fn glyphs(&self) -> Slice2<T> {
        self.glyphs.as_slice2()
    }

    pub fn glyph_size(&self) -> (u32, u32) {
        (self.glyph_w, self.glyph_h)
    }
}

impl<T> Font<T> {
    /// Renders ("bakes") the byte string into a buffer.
    pub fn bake(s: &[u8], font: &Font<T>, mut dest: impl AsMutSlice2<T>)
    where
        T: Copy + Default,
    {
        let mut dest = dest.as_mut_slice2();
        let lines = s.split(|&c| c == b'\n');
        let (num_rows, num_cols) = lines
            .clone()
            .fold((0, 0), |(nr, nc), row| (nr + 1, nc.max(row.len() as u32)));
        if num_rows == 0 || num_cols == 0 {
            return;
        }
        let (gw, gh) = font.glyph_size();

        let (mut x, mut y) = (0, 0);
        for line in lines {
            for c in line {
                let mut dest = dest.slice_mut((x..x + gw, y..y + gw));
                dest.copy_from(font.glyph(*c));
                x += gw;
            }
            (x, y) = (0, y + gh);
        }
    }
}

pub struct Table {
    cols: Vec<Col>,
    cells: Vec<Cell>,
    borders: &'static str,
}

pub struct Col {
    w: usize,
}

pub struct Cell {
    _h: usize,
}

impl Display for Table {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let nrows = self.cells.len() / self.cols.len();
        let brd: Vec<_> = self.borders.chars().collect();

        let mut b = &brd[0..4];
        let mut left = b[0];
        for _r in 0..nrows {
            // horizontal border
            for c in &self.cols {
                write!(f, "{left}{}", b[1].to_string().repeat(c.w))?;
                left = b[2];
            }
            writeln!(f, "{}", b[3])?;

            // content and vertical borders
            b = &brd[4..8];
            left = b[0];
            for c in &self.cols {
                write!(f, "{left}{}", b[1].to_string().repeat(c.w))?;
                left = b[2];
            }
            writeln!(f, "{}", b[3])?;
            b = &brd[8..12];
            left = b[0];
        }

        // bottom border
        b = &brd[12..16];
        left = b[0];
        for c in &self.cols {
            write!(f, "{left}{}", b[1].to_string().repeat(c.w))?;
            left = b[2];
        }
        write!(f, "{}", b[3])
    }
}

#[cfg(test)]
mod tests {
    use std::println;
    use std::vec;

    use super::*;

    #[test]
    fn table_display() {
        let t = Table {
            cols: vec![Col { w: 4 }, Col { w: 6 }, Col { w: 3 }],
            cells: vec![
                Cell { h: 1 },
                Cell { h: 1 },
                Cell { h: 1 },
                Cell { h: 1 },
                Cell { h: 1 },
                Cell { h: 1 },
            ],
            borders: "┌─┬┐\
                      | ||\
                      ├─┼┤\
                      └─┴┘",
        };

        println!("{}", t);
    }
}

/// Utilities for working with the original IBM PC code page 437 encoding.
pub mod cp437 {
    use alloc::vec::Vec;

    /// Converts each character read from the iterator to code page 437 and
    /// returns an iterator yielding the resulting code points.
    ///
    /// For any character that does not have a CP437 equivalent, the replacement
    /// function `repl` is called. If `repl` returns `None`, that character is
    /// skipped; otherwise, the returned code point is output.
    pub fn encode(
        it: impl IntoIterator<Item = char>,
        repl: impl Fn(char) -> Option<u8> + 'static,
    ) -> impl Iterator<Item = u8> {
        it.into_iter()
            .flat_map(move |c| encode_char(c).or_else(|| repl(c)))
    }

    /// Converts each character in the string to code page 437 and
    /// returns a vector of the resulting code points.
    ///
    /// For any character that does not have a CP437 equivalent, the replacement
    /// function `repl` is called. If `repl` returns `None`, that character is
    /// skipped; otherwise, the returned code point is output.
    pub fn encode_str(
        s: &str,
        repl: impl Fn(char) -> Option<u8> + 'static,
    ) -> Vec<u8> {
        encode(s.chars(), repl).collect()
    }

    /// Returns the code point equivalent to `c` in code page 437,
    /// or `None` if there is no equivalent.
    pub fn encode_char(c: char) -> Option<u8> {
        let res = match c {
            '\x00'..='\x7f' => c as u8,
            '☺' => 0x01, // U+263A
            '☻' => 0x02, // U+263B
            '♥' => 0x03, // U+2665
            '♦' => 0x04, // U+2666
            '♣' => 0x05, // U+2663
            '♠' => 0x06, // U+2660
            '•' => 0x07, // U+2022
            '◘' => 0x08, // U+25D8
            '○' => 0x09, // U+25CB
            '◙' => 0x0A, // U+25D9
            '♂' => 0x0B, // U+2642
            '♀' => 0x0C, // U+2640
            '♪' => 0x0D, // U+266A
            '♫' => 0x0E, // U+266B
            '☼' => 0x0F, // U+263C

            '►' => 0x10, // U+25BA
            '◄' => 0x11, // U+25C4
            '↕' => 0x12, // U+2195
            '‼' => 0x13, // U+203C
            '¶' => 0x14, // U+00B6
            '§' => 0x15, // U+00A7
            '▬' => 0x16, // U+25AC
            '↨' => 0x17, // U+21A8
            '↑' => 0x18, // U+2191
            '↓' => 0x19, // U+2193
            '→' => 0x1A, // U+2192
            '←' => 0x1B, // U+2190
            '∟' => 0x1C, // U+221F
            '↔' => 0x1D, // U+2194
            '▲' => 0x1E, // U+25B2
            '▼' => 0x1F, // U+25BC

            '⌂' => 0x7F, // U+2302

            'Ç' => 0x80, // U+00C7
            'ü' => 0x81, // U+00FC
            'é' => 0x82, // U+00E9
            'â' => 0x83, // U+00E2
            'ä' => 0x84, // U+00E4
            'à' => 0x85, // U+00E0
            'å' => 0x86, // U+00E5
            'ç' => 0x87, // U+00E7
            'ê' => 0x88, // U+00EA
            'ë' => 0x89, // U+00EB
            'è' => 0x8A, // U+00E8
            'ï' => 0x8B, // U+00EF
            'î' => 0x8C, // U+00EE
            'ì' => 0x8D, // U+00EC
            'Ä' => 0x8E, // U+00C4
            'Å' => 0x8F, // U+00C5

            'É' => 0x90, // U+00C9
            'æ' => 0x91, // U+00E6
            'Æ' => 0x92, // U+00C6
            'ô' => 0x93, // U+00F4
            'ö' => 0x94, // U+00F6
            'ò' => 0x95, // U+00F2
            'û' => 0x96, // U+00FB
            'ù' => 0x97, // U+00F9
            'ÿ' => 0x98, // U+00FF
            'Ö' => 0x99, // U+00D6
            'Ü' => 0x9A, // U+00DC
            '¢' => 0x9B, // U+00A2
            '£' => 0x9C, // U+00A3
            '¥' => 0x9D, // U+00A5
            '₧' => 0x9E, // U+20A7
            'ƒ' => 0x9F, // U+0192

            'á' => 0xA0, // U+00E1
            'í' => 0xA1, // U+00ED
            'ó' => 0xA2, // U+00F3
            'ú' => 0xA3, // U+00FA
            'ñ' => 0xA4, // U+00F1
            'Ñ' => 0xA5, // U+00D1
            'ª' => 0xA6, // U+00AA
            'º' => 0xA7, // U+00BA
            '¿' => 0xA8, // U+00BF
            '⌐' => 0xA9, // U+2310
            '¬' => 0xAA, // U+00AC
            '½' => 0xAB, // U+00BD
            '¼' => 0xAC, // U+00BC
            '¡' => 0xAD, // U+00A1
            '«' => 0xAE, // U+00AB
            '»' => 0xAF, // U+00BB

            '░' => 0xB0, // U+2591
            '▒' => 0xB1, // U+2592
            '▓' => 0xB2, // U+2593
            '│' => 0xB3, // U+2502
            '┤' => 0xB4, // U+2524
            '╡' => 0xB5, // U+2561
            '╢' => 0xB6, // U+2562
            '╖' => 0xB7, // U+2556
            '╕' => 0xB8, // U+2555
            '╣' => 0xB9, // U+2563
            '║' => 0xBA, // U+2551
            '╗' => 0xBB, // U+2557
            '╝' => 0xBC, // U+255D
            '╜' => 0xBD, // U+255C
            '╛' => 0xBE, // U+255B
            '┐' => 0xBF, // U+2510

            '└' => 0xC0, // U+2514
            '┴' => 0xC1, // U+2534
            '┬' => 0xC2, // U+252C
            '├' => 0xC3, // U+251C
            '─' => 0xC4, // U+2500
            '┼' => 0xC5, // U+253C
            '╞' => 0xC6, // U+255E
            '╟' => 0xC7, // U+255F
            '╚' => 0xC8, // U+255A
            '╔' => 0xC9, // U+2554
            '╩' => 0xCA, // U+2569
            '╦' => 0xCB, // U+2566
            '╠' => 0xCC, // U+2560
            '═' => 0xCD, // U+2550
            '╬' => 0xCE, // U+256C
            '╧' => 0xCF, // U+2567

            '╨' => 0xD0, // U+2568
            '╤' => 0xD1, // U+2564
            '╥' => 0xD2, // U+2565
            '╙' => 0xD3, // U+2559
            '╘' => 0xD4, // U+2558
            '╒' => 0xD5, // U+2552
            '╓' => 0xD6, // U+2553
            '╫' => 0xD7, // U+256B
            '╪' => 0xD8, // U+256A
            '┘' => 0xD9, // U+2518
            '┌' => 0xDA, // U+250C
            '█' => 0xDB, // U+2588
            '▄' => 0xDC, // U+2584
            '▌' => 0xDD, // U+258C
            '▐' => 0xDE, // U+2590
            '▀' => 0xDF, // U+2580

            'α' => 0xE0, // U+03B1
            'ß' => 0xE1, // U+00DF
            'Γ' => 0xE2, // U+0393
            'π' => 0xE3, // U+03C0
            'Σ' => 0xE4, // U+03A3
            'σ' => 0xE5, // U+03C3
            'µ' => 0xE6, // U+00B5
            'τ' => 0xE7, // U+03C4
            'Φ' => 0xE8, // U+03A6
            'Θ' => 0xE9, // U+0398
            'Ω' => 0xEA, // U+03A9
            'δ' => 0xEB, // U+03B4
            '∞' => 0xEC, // U+221E
            'φ' => 0xED, // U+03C6
            'ε' => 0xEE, // U+03B5
            '∩' => 0xEF, // U+2229

            '≡' => 0xF0, // U+2261
            '±' => 0xF1, // U+00B1
            '≥' => 0xF2, // U+2265
            '≤' => 0xF3, // U+2264
            '⌠' => 0xF4, // U+2320
            '⌡' => 0xF5, // U+2321
            '÷' => 0xF6, // U+00F7
            '≈' => 0xF7, // U+2248
            '°' => 0xF8, // U+00B0
            '∙' => 0xF9, // U+2219
            '·' => 0xFA, // U+00B7
            '√' => 0xFB, // U+221A
            'ⁿ' => 0xFC, // U+207F
            '²' => 0xFD, // U+00B2
            '■' => 0xFE, // U+25A0
            ' ' => 0xFF, // U+00A0

            _ => return None,
        };
        Some(res)
    }
}
