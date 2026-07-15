import { useMemo } from 'react';

type AnsiSegment = {
  text: string;
  style: React.CSSProperties;
};

const ANSI_COLORS: Record<number, string> = {
  30: '#000000',
  31: '#cd3131',
  32: '#0dbc79',
  33: '#e5e510',
  34: '#2472c8',
  35: '#bc3fbc',
  36: '#11a8cd',
  37: '#e5e5e5',

  90: '#666666',
  91: '#f14c4c',
  92: '#23d18b',
  93: '#f5f543',
  94: '#3b8eea',
  95: '#d670d6',
  96: '#29b8db',
  97: '#ffffff',
};

const ANSI_BACKGROUND_COLORS: Record<number, string> = {
  40: '#000000',
  41: '#cd3131',
  42: '#0dbc79',
  43: '#e5e510',
  44: '#2472c8',
  45: '#bc3fbc',
  46: '#11a8cd',
  47: '#e5e5e5',

  100: '#666666',
  101: '#f14c4c',
  102: '#23d18b',
  103: '#f5f543',
  104: '#3b8eea',
  105: '#d670d6',
  106: '#29b8db',
  107: '#ffffff',
};

export function xterm256ToHex(code: number): string {
  if (code < 16) {
    const baseColors = [
      '#000000',
      '#800000',
      '#008000',
      '#808000',
      '#000080',
      '#800080',
      '#008080',
      '#c0c0c0',
      '#808080',
      '#ff0000',
      '#00ff00',
      '#ffff00',
      '#0000ff',
      '#ff00ff',
      '#00ffff',
      '#ffffff',
    ];

    return baseColors[code] ?? '#ffffff';
  }

  if (code >= 16 && code <= 231) {
    const n = code - 16;
    const r = Math.floor(n / 36);
    const g = Math.floor((n % 36) / 6);
    const b = n % 6;

    const convert = (value: number) => (value === 0 ? 0 : 55 + value * 40);

    return `rgb(${convert(r)}, ${convert(g)}, ${convert(b)})`;
  }

  if (code >= 232 && code <= 255) {
    const level = 8 + (code - 232) * 10;
    return `rgb(${level}, ${level}, ${level})`;
  }

  return '#ffffff';
}

export function parseAnsi(input: string): AnsiSegment[] {
  const segments: AnsiSegment[] = [];
  const ESC = String.fromCharCode(27);
  const ansiRegex = new RegExp(`${ESC}\\[([0-9;]*)m`, 'g');

  let currentStyle: React.CSSProperties = {};
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  const pushText = (text: string) => {
    if (text) {
      segments.push({
        text,
        style: { ...currentStyle },
      });
    }
  };

  while ((match = ansiRegex.exec(input)) !== null) {
    pushText(input.slice(lastIndex, match.index));

    const rawCodes = match[1];
    const codes = rawCodes === '' ? [0] : rawCodes.split(';').map(Number);

    for (let i = 0; i < codes.length; i++) {
      const code = codes[i];

      if (code === 0) {
        currentStyle = {};
      } else if (code === 1) {
        currentStyle.fontWeight = 'bold';
      } else if (code === 2) {
        currentStyle.opacity = 0.65;
      } else if (code === 3) {
        currentStyle.fontStyle = 'italic';
      } else if (code === 4) {
        currentStyle.textDecoration = 'underline';
      } else if (code === 22) {
        delete currentStyle.fontWeight;
        delete currentStyle.opacity;
      } else if (code === 23) {
        delete currentStyle.fontStyle;
      } else if (code === 24) {
        delete currentStyle.textDecoration;
      } else if (code === 39) {
        delete currentStyle.color;
      } else if (code === 49) {
        delete currentStyle.backgroundColor;
      } else if (ANSI_COLORS[code]) {
        currentStyle.color = ANSI_COLORS[code];
      } else if (ANSI_BACKGROUND_COLORS[code]) {
        currentStyle.backgroundColor = ANSI_BACKGROUND_COLORS[code];
      } else if (code === 38 || code === 48) {
        const isForeground = code === 38;
        const mode = codes[i + 1];

        // 256-color mode: ESC[38;5;{n}m or ESC[48;5;{n}m
        if (mode === 5 && typeof codes[i + 2] === 'number') {
          const color = xterm256ToHex(codes[i + 2]);

          if (isForeground) {
            currentStyle.color = color;
          } else {
            currentStyle.backgroundColor = color;
          }

          i += 2;
        }

        // True-color mode: ESC[38;2;r;g;bm or ESC[48;2;r;g;bm
        else if (mode === 2 && typeof codes[i + 2] === 'number' && typeof codes[i + 3] === 'number' && typeof codes[i + 4] === 'number') {
          const r = codes[i + 2];
          const g = codes[i + 3];
          const b = codes[i + 4];
          const color = `rgb(${r}, ${g}, ${b})`;

          if (isForeground) {
            currentStyle.color = color;
          } else {
            currentStyle.backgroundColor = color;
          }

          i += 4;
        }
      }
    }

    lastIndex = ansiRegex.lastIndex;
  }

  pushText(input.slice(lastIndex));

  return segments;
}

type AnsiTextProps = {
  text: string;
};

export function AnsiText({ text }: AnsiTextProps) {
  const segments = useMemo(() => parseAnsi(text), [text]);

  return (
    <>
      {segments.map((segment, index) => (
        <span key={index} style={segment.style}>
          {segment.text}
        </span>
      ))}
    </>
  );
}
