import { describe, it, expect } from 'vitest';

// project imports
import {
  decodeText,
  detectRenderKind,
  editorFormatHint,
  extensionOf,
  formatFromFileName,
  imageMimeForName,
  isImageFile,
  isJsonText,
  looksLikeText,
  textOf,
} from './detect';
import { RenderKind } from './types';
import { FormatType } from '@utilities/rules/types';

const enc = (s: string) => new TextEncoder().encode(s).buffer;
const bin = (...bytes: number[]) => new Uint8Array(bytes).buffer;

describe('looksLikeText', () => {
  it('treats ascii as text', () => {
    expect(looksLikeText(enc('hello world\n'))).toBe(true);
  });
  it('treats a NUL byte as binary', () => {
    expect(looksLikeText(bin(0x68, 0x00, 0x69))).toBe(false);
  });
  it('treats mostly non-printable bytes as binary', () => {
    expect(looksLikeText(bin(0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x0b))).toBe(false);
  });
  it('treats empty input as text', () => {
    expect(looksLikeText(new ArrayBuffer(0))).toBe(true);
  });
});

describe('decodeText', () => {
  it('round-trips utf-8', () => {
    expect(decodeText(enc('café ☕'))).toBe('café ☕');
  });
});

describe('textOf', () => {
  it('prefers the pre-decoded text when present', () => {
    expect(textOf({ bytes: enc('from-bytes'), text: 'from-text' })).toBe('from-text');
  });
  it('decodes the bytes when no text is present', () => {
    expect(textOf({ bytes: enc('café ☕') })).toBe('café ☕');
  });
  it('returns an empty-string text verbatim rather than decoding bytes', () => {
    expect(textOf({ bytes: enc('from-bytes'), text: '' })).toBe('');
  });
});

describe('detectRenderKind', () => {
  it('detects yara by extension', () => {
    expect(detectRenderKind('rules.yar', enc('rule x {}'))).toBe(RenderKind.Yara);
    expect(detectRenderKind('rules.yara', enc('anything'))).toBe(RenderKind.Yara);
  });

  it('detects json by extension', () => {
    expect(detectRenderKind('out.json', enc('{"a":1}'))).toBe(RenderKind.Json);
  });

  it('classifies binary content as hex regardless of extension', () => {
    expect(detectRenderKind('data.json', bin(0x00, 0x01, 0x02, 0xff))).toBe(RenderKind.Hex);
    expect(detectRenderKind('mystery', bin(0xde, 0xad, 0xbe, 0xef, 0x00))).toBe(RenderKind.Hex);
  });

  it('detects code by extension', () => {
    expect(detectRenderKind('config.yaml', enc('a: 1'))).toBe(RenderKind.Code);
    expect(detectRenderKind('script.py', enc('print(1)'))).toBe(RenderKind.Code);
  });

  it('detects plain text by extension', () => {
    expect(detectRenderKind('notes.txt', enc('just text'))).toBe(RenderKind.Text);
    expect(detectRenderKind('server.log', enc('line 1'))).toBe(RenderKind.Text);
    expect(detectRenderKind('data.csv', enc('a,b,c'))).toBe(RenderKind.Text);
  });

  it('detects markdown by extension', () => {
    expect(detectRenderKind('README.md', enc('# Title'))).toBe(RenderKind.Markdown);
    expect(detectRenderKind('doc.markdown', enc('text'))).toBe(RenderKind.Markdown);
  });

  it('sniffs json content when extension is missing', () => {
    expect(detectRenderKind('', enc('  {"a": [1,2,3]}  '))).toBe(RenderKind.Json);
    expect(detectRenderKind('blob', enc('[1,2,3]'))).toBe(RenderKind.Json);
  });

  it('sniffs yara content without an extension', () => {
    expect(detectRenderKind('blob', enc('import "pe"\nrule Evil {\n  condition: true\n}'))).toBe(RenderKind.Yara);
  });

  it('sniffs yara rules with private/global modifiers', () => {
    expect(detectRenderKind('blob', enc('private rule Hidden {\n  condition: true\n}'))).toBe(RenderKind.Yara);
    expect(detectRenderKind('blob', enc('global private rule Both {\n  condition: true\n}'))).toBe(RenderKind.Yara);
  });

  it('falls back to text for unknown content', () => {
    expect(detectRenderKind('', enc('hello there, not json'))).toBe(RenderKind.Text);
  });

  it('detects images by extension', () => {
    expect(detectRenderKind('logo.png', bin(0x00, 0x01))).toBe(RenderKind.Image);
    expect(detectRenderKind('photo.JPG', bin(0x00, 0x01))).toBe(RenderKind.Image);
    expect(detectRenderKind('icon.svg', enc('<svg></svg>'))).toBe(RenderKind.Image);
  });

  it('detects images by magic bytes even with a wrong/missing extension', () => {
    // PNG signature
    expect(detectRenderKind('mystery', bin(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a))).toBe(RenderKind.Image);
    // JPEG signature with a .bin extension
    expect(detectRenderKind('photo.bin', bin(0xff, 0xd8, 0xff, 0xe0))).toBe(RenderKind.Image);
    // GIF signature
    expect(detectRenderKind('', bin(0x47, 0x49, 0x46, 0x38, 0x39, 0x61))).toBe(RenderKind.Image);
  });
});

describe('extensionOf', () => {
  it('returns the lower-cased extension without the dot', () => {
    expect(extensionOf('IMG.PNG')).toBe('png');
    expect(extensionOf('archive.tar.gz')).toBe('gz');
  });
  it('returns empty string for no-dot, trailing-dot, or undefined names', () => {
    expect(extensionOf('noext')).toBe('');
    expect(extensionOf('trailing.')).toBe('');
    expect(extensionOf(undefined)).toBe('');
  });
});

describe('formatFromFileName', () => {
  it('maps json/yara extensions and defaults everything else to YAML', () => {
    expect(formatFromFileName('x.json')).toBe(FormatType.JSON);
    expect(formatFromFileName('x.yar')).toBe(FormatType.YARA);
    expect(formatFromFileName('x.yara')).toBe(FormatType.YARA);
    expect(formatFromFileName('x.txt')).toBe(FormatType.YAML);
    expect(formatFromFileName(undefined)).toBe(FormatType.YAML);
  });
});

describe('imageMimeForName', () => {
  it('maps known image extensions to MIME types', () => {
    expect(imageMimeForName('a.svgz')).toBe('image/svg+xml');
    expect(imageMimeForName('a.jpg')).toBe('image/jpeg');
    expect(imageMimeForName('a.tif')).toBe('image/tiff');
  });
  it('returns undefined for unknown extensions', () => {
    expect(imageMimeForName('a.txt')).toBeUndefined();
    expect(imageMimeForName(undefined)).toBeUndefined();
  });
});

describe('isImageFile', () => {
  it('detects images by extension', () => {
    expect(isImageFile('logo.png', bin(0x00))).toBe(true);
  });
  it('detects WEBP by RIFF magic', () => {
    // "RIFF" .... "WEBP"
    expect(isImageFile('x', bin(0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x50))).toBe(true);
  });
  it('detects AVIF by ftyp brand', () => {
    expect(isImageFile('x', bin(0, 0, 0, 0, 0x66, 0x74, 0x79, 0x70, 0x61, 0x76, 0x69, 0x66))).toBe(true);
  });
  it('detects TIFF by magic', () => {
    expect(isImageFile('x', bin(0x49, 0x49, 0x2a, 0x00))).toBe(true);
  });
  it('returns false for non-image content with no image extension', () => {
    expect(isImageFile('data.txt', bin(0x68, 0x69))).toBe(false);
  });
});

describe('isJsonText', () => {
  it('accepts objects and arrays with leading whitespace', () => {
    expect(isJsonText('  {"a":1}')).toBe(true);
    expect(isJsonText('[1,2,3]')).toBe(true);
  });
  it('rejects empty, plain text, and invalid JSON that starts with a brace', () => {
    expect(isJsonText('')).toBe(false);
    expect(isJsonText('plain text')).toBe(false);
    expect(isJsonText('{not valid}')).toBe(false);
  });
});

describe('editorFormatHint', () => {
  it('returns JSON for .json and for content-detected JSON', () => {
    expect(editorFormatHint('out.json', enc('{"a":1}'))).toBe(FormatType.JSON);
    expect(editorFormatHint('blob', enc('{"a":1}'))).toBe(FormatType.JSON);
  });

  it('returns YARA for .yar/.yara', () => {
    expect(editorFormatHint('rules.yar', enc('rule x {}'))).toBe(FormatType.YARA);
  });

  it('returns YAML for .yaml and as the default for plain/code files', () => {
    expect(editorFormatHint('config.yaml', enc('a: 1'))).toBe(FormatType.YAML);
    expect(editorFormatHint('notes.txt', enc('hello'))).toBe(FormatType.YAML);
  });
});
