import { describe, expect, it } from 'vitest';
import { parseAnsi, xterm256ToHex } from './AnsiText';

const ESC = String.fromCharCode(27);

describe('xterm256ToHex', () => {
  it('converts basic 256-color palette values', () => {
    expect(xterm256ToHex(0)).toBe('#000000');
    expect(xterm256ToHex(1)).toBe('#800000');
    expect(xterm256ToHex(2)).toBe('#008000');
    expect(xterm256ToHex(15)).toBe('#ffffff');
  });

  it('converts 256-color cube values', () => {
    expect(xterm256ToHex(16)).toBe('rgb(0, 0, 0)');
    expect(xterm256ToHex(196)).toBe('rgb(255, 0, 0)');
    expect(xterm256ToHex(46)).toBe('rgb(0, 255, 0)');
    expect(xterm256ToHex(21)).toBe('rgb(0, 0, 255)');
    expect(xterm256ToHex(231)).toBe('rgb(255, 255, 255)');
  });

  it('converts grayscale 256-color values', () => {
    expect(xterm256ToHex(232)).toBe('rgb(8, 8, 8)');
    expect(xterm256ToHex(255)).toBe('rgb(238, 238, 238)');
  });

  it('returns a fallback for invalid values', () => {
    expect(xterm256ToHex(-1)).toBe('#ffffff');
    expect(xterm256ToHex(999)).toBe('#ffffff');
  });
});

describe('parseAnsi', () => {
  it('returns plain text as a single unstyled segment', () => {
    expect(parseAnsi('hello world')).toEqual([
      {
        text: 'hello world',
        style: {},
      },
    ]);
  });

  it('parses green foreground text', () => {
    expect(parseAnsi(`${ESC}[32mINFO${ESC}[0m`)).toEqual([
      {
        text: 'INFO',
        style: {
          color: '#0dbc79',
        },
      },
    ]);
  });

  it('resets style after reset code', () => {
    expect(parseAnsi(`${ESC}[32mINFO${ESC}[0m normal`)).toEqual([
      {
        text: 'INFO',
        style: {
          color: '#0dbc79',
        },
      },
      {
        text: ' normal',
        style: {},
      },
    ]);
  });

  it('parses dim text', () => {
    expect(parseAnsi(`${ESC}[2mdim text${ESC}[0m`)).toEqual([
      {
        text: 'dim text',
        style: {
          opacity: 0.65,
        },
      },
    ]);
  });

  it('parses bold text', () => {
    expect(parseAnsi(`${ESC}[1mbold text${ESC}[0m`)).toEqual([
      {
        text: 'bold text',
        style: {
          fontWeight: 'bold',
        },
      },
    ]);
  });

  it('parses italic text', () => {
    expect(parseAnsi(`${ESC}[3mitalic text${ESC}[0m`)).toEqual([
      {
        text: 'italic text',
        style: {
          fontStyle: 'italic',
        },
      },
    ]);
  });

  it('parses underlined text', () => {
    expect(parseAnsi(`${ESC}[4munderlined text${ESC}[0m`)).toEqual([
      {
        text: 'underlined text',
        style: {
          textDecoration: 'underline',
        },
      },
    ]);
  });

  it('parses foreground and background colors', () => {
    expect(parseAnsi(`${ESC}[31mred${ESC}[0m ${ESC}[44mblue bg${ESC}[0m`)).toEqual([
      {
        text: 'red',
        style: {
          color: '#cd3131',
        },
      },
      {
        text: ' ',
        style: {},
      },
      {
        text: 'blue bg',
        style: {
          backgroundColor: '#2472c8',
        },
      },
    ]);
  });

  it('parses bright foreground colors', () => {
    expect(parseAnsi(`${ESC}[91mbright red${ESC}[0m`)).toEqual([
      {
        text: 'bright red',
        style: {
          color: '#f14c4c',
        },
      },
    ]);
  });

  it('parses bright background colors', () => {
    expect(parseAnsi(`${ESC}[101mbright red bg${ESC}[0m`)).toEqual([
      {
        text: 'bright red bg',
        style: {
          backgroundColor: '#f14c4c',
        },
      },
    ]);
  });

  it('parses combined ANSI codes', () => {
    expect(parseAnsi(`${ESC}[1;32mbold green${ESC}[0m`)).toEqual([
      {
        text: 'bold green',
        style: {
          fontWeight: 'bold',
          color: '#0dbc79',
        },
      },
    ]);
  });

  it('parses 256-color foreground codes', () => {
    expect(parseAnsi(`${ESC}[38;5;196mred 256${ESC}[0m`)).toEqual([
      {
        text: 'red 256',
        style: {
          color: 'rgb(255, 0, 0)',
        },
      },
    ]);
  });

  it('parses 256-color background codes', () => {
    expect(parseAnsi(`${ESC}[48;5;196mred bg 256${ESC}[0m`)).toEqual([
      {
        text: 'red bg 256',
        style: {
          backgroundColor: 'rgb(255, 0, 0)',
        },
      },
    ]);
  });

  it('parses true-color foreground codes', () => {
    expect(parseAnsi(`${ESC}[38;2;10;20;30mtrue color${ESC}[0m`)).toEqual([
      {
        text: 'true color',
        style: {
          color: 'rgb(10, 20, 30)',
        },
      },
    ]);
  });

  it('parses true-color background codes', () => {
    expect(parseAnsi(`${ESC}[48;2;10;20;30mtrue bg${ESC}[0m`)).toEqual([
      {
        text: 'true bg',
        style: {
          backgroundColor: 'rgb(10, 20, 30)',
        },
      },
    ]);
  });

  it('handles foreground color reset code 39', () => {
    expect(parseAnsi(`${ESC}[32mgreen${ESC}[39m default`)).toEqual([
      {
        text: 'green',
        style: {
          color: '#0dbc79',
        },
      },
      {
        text: ' default',
        style: {},
      },
    ]);
  });

  it('handles background color reset code 49', () => {
    expect(parseAnsi(`${ESC}[41mred bg${ESC}[49m default`)).toEqual([
      {
        text: 'red bg',
        style: {
          backgroundColor: '#cd3131',
        },
      },
      {
        text: ' default',
        style: {},
      },
    ]);
  });

  it('handles bold and dim reset code 22', () => {
    expect(parseAnsi(`${ESC}[1;2mbold dim${ESC}[22m normal`)).toEqual([
      {
        text: 'bold dim',
        style: {
          fontWeight: 'bold',
          opacity: 0.65,
        },
      },
      {
        text: ' normal',
        style: {},
      },
    ]);
  });

  it('handles italic reset code 23', () => {
    expect(parseAnsi(`${ESC}[3mitalic${ESC}[23m normal`)).toEqual([
      {
        text: 'italic',
        style: {
          fontStyle: 'italic',
        },
      },
      {
        text: ' normal',
        style: {},
      },
    ]);
  });

  it('handles underline reset code 24', () => {
    expect(parseAnsi(`${ESC}[4munderlined${ESC}[24m normal`)).toEqual([
      {
        text: 'underlined',
        style: {
          textDecoration: 'underline',
        },
      },
      {
        text: ' normal',
        style: {},
      },
    ]);
  });

  it('handles empty reset sequence ESC[m', () => {
    expect(parseAnsi(`${ESC}[32mgreen${ESC}[m normal`)).toEqual([
      {
        text: 'green',
        style: {
          color: '#0dbc79',
        },
      },
      {
        text: ' normal',
        style: {},
      },
    ]);
  });

  it('parses the provided log example', () => {
    const line =
      `${ESC}[2m2026-07-15T00:45:01.463854Z${ESC}[0m ` +
      `${ESC}[32m INFO${ESC}[0m ` +
      `${ESC}[2mghidra_cli::ghidra::bridge${ESC}[0m` +
      `${ESC}[2m:${ESC}[0m ` +
      `[Ghidra import stderr] openjdk version "21.0.11" 2026-04-21`;

    expect(parseAnsi(line)).toEqual([
      {
        text: '2026-07-15T00:45:01.463854Z',
        style: {
          opacity: 0.65,
        },
      },
      {
        text: ' ',
        style: {},
      },
      {
        text: ' INFO',
        style: {
          color: '#0dbc79',
        },
      },
      {
        text: ' ',
        style: {},
      },
      {
        text: 'ghidra_cli::ghidra::bridge',
        style: {
          opacity: 0.65,
        },
      },
      {
        text: ':',
        style: {
          opacity: 0.65,
        },
      },
      {
        text: ' [Ghidra import stderr] openjdk version "21.0.11" 2026-04-21',
        style: {},
      },
    ]);
  });
});
