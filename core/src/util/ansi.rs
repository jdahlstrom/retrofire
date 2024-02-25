use core::fmt::{self, Display, Formatter};

use std::print;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Ansi {
    TerminalReset,

    CursorSave,
    CursorRestore,

    CursorUp(u16),
    CursorDown(u16),
    CursorFwd(u16),
    CursorBack(u16),

    CursorPos(u16, u16),

    Graphic(Attrib),
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Attrib {
    Reset,
    Foreground(Color),
    Background(Color),
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Color {
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,

    BrightBlack,
    BrightRed,
    BrightGreen,
    BrightYellow,
    BrightBlue,
    BrightMagenta,
    BrightCyan,
    BrightWhite,

    EightBit(u8),
    TrueColor(u8, u8, u8),
}

impl Display for Ansi {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use Ansi::*;
        use Attrib::*;
        match self {
            TerminalReset => write!(f, "\x1Bc"),
            CursorSave => write!(f, "\x1B7"),
            CursorRestore => write!(f, "\x1B8"),
            CursorUp(n) => write!(f, "\x1B[{n}A"),
            CursorDown(n) => write!(f, "\x1B[{n}B"),
            CursorFwd(n) => write!(f, "\x1B[{n}C"),
            CursorBack(n) => write!(f, "\x1B[{n}D"),
            CursorPos(row, col) => write!(f, "\x1B[{row};{col}H"),
            Graphic(Reset) => write!(f, "\x1B[0m"),
            Graphic(Foreground(col)) => write!(f, "\x1B[38;{col}m"),
            Graphic(Background(col)) => write!(f, "\x1B[48;{col}m"),
        }
    }
}
impl Display for Attrib {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        todo!()
    }
}
impl Display for Color {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        use Color::*;
        match self {
            Black => write!(f, "5;0"),
            Red => write!(f, "5;1"),
            Green => write!(f, "5;2"),
            Yellow => write!(f, "5;3"),
            Blue => write!(f, "5;4"),
            Magenta => write!(f, "5;5"),
            Cyan => write!(f, "5;6"),
            White => write!(f, "5;7"),

            BrightBlack => write!(f, "5;8"),
            BrightRed => write!(f, "5;9"),
            BrightGreen => write!(f, "5;10"),
            BrightYellow => write!(f, "5;11"),
            BrightBlue => write!(f, "5;12"),
            BrightMagenta => write!(f, "5;13"),
            BrightCyan => write!(f, "5;14"),
            BrightWhite => write!(f, "5;15"),

            EightBit(c) => write!(f, "5;{c}"),
            TrueColor(r, g, b) => write!(f, "2;{r};{g};{b}"),
        }
    }
}

impl Ansi {
    pub fn start() -> Sequence {
        Sequence
    }
}

pub struct Sequence;

impl Sequence {
    pub fn cursor_save(self) -> Self {
        print!("{}", Ansi::CursorSave);
        self
    }
    pub fn cursor_restore(self) -> Self {
        print!("{}", Ansi::CursorRestore);
        self
    }
    pub fn cursor_up(self, n: u16) -> Self {
        print!("{}", Ansi::CursorUp(n));
        self
    }
    pub fn cursor_down(self, n: u16) -> Self {
        print!("{}", Ansi::CursorDown(n));
        self
    }
    pub fn cursor_fwd(self, n: u16) -> Self {
        print!("{}", Ansi::CursorFwd(n));
        self
    }
    pub fn cursor_back(self, n: u16) -> Self {
        print!("{}", Ansi::CursorBack(n));
        self
    }
    pub fn cursor_pos(self, row: u16, col: u16) -> Self {
        print!("{}", Ansi::CursorPos(row, col));
        self
    }
    pub fn fg_color(self, c: Color) -> Self {
        print!("{}", Ansi::Graphic(Attrib::Foreground(c)));
        self
    }
    pub fn bg_color(self, c: Color) -> Self {
        print!("{}", Ansi::Graphic(Attrib::Background(c)));
        self
    }
    pub fn write(self, str: &str) -> Self {
        print!("{}", str);
        self
    }
    pub fn reset(self) -> Self {
        print!("{}", Ansi::Graphic(Attrib::Reset));
        self
    }
}
