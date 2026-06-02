import { describe, it, expect, vi } from 'vitest';

// project imports
import { buildUploadFormBase, appendOriginToForm } from './originValidation';
import { DEFAULT_ORIGIN_STATE, OriginState } from './types';

vi.mock('@utilities/tags', () => ({
  hasInvalidTags: (tags: { key: string; value: string }[]) => tags.some((t) => t.key === 'INVALID'),
}));

function originWith(patch: Partial<OriginState>): OriginState {
  return { ...structuredClone(DEFAULT_ORIGIN_STATE), ...patch };
}

describe('buildUploadFormBase', () => {
  it('returns form when groups are provided', () => {
    const result = buildUploadFormBase('', ['group1'], [], []);
    expect('form' in result).toBe(true);
  });

  it('returns error when no groups selected', () => {
    const result = buildUploadFormBase('', [], [], []);
    expect('errors' in result).toBe(true);
    if ('errors' in result) {
      expect(result.errors[0]).toMatch(/group/i);
    }
  });

  it('appends description when provided', () => {
    const result = buildUploadFormBase('test desc', ['g1'], [], []);
    expect('form' in result).toBe(true);
    if ('form' in result) {
      expect(result.form.get('description')).toBe('test desc');
    }
  });

  it('does not append description when empty', () => {
    const result = buildUploadFormBase('', ['g1'], [], []);
    expect('form' in result).toBe(true);
    if ('form' in result) {
      expect(result.form.has('description')).toBe(false);
    }
  });

  it('appends multiple groups', () => {
    const result = buildUploadFormBase('', ['g1', 'g2'], [], []);
    expect('form' in result).toBe(true);
    if ('form' in result) {
      expect(result.form.getAll('groups')).toEqual(['g1', 'g2']);
    }
  });

  it('appends TLP tags', () => {
    const tlp = [{ key: 'TLP', value: 'GREEN' }];
    const result = buildUploadFormBase('', ['g1'], [], tlp);
    expect('form' in result).toBe(true);
    if ('form' in result) {
      expect(result.form.get('tags[TLP]')).toBe('GREEN');
    }
  });

  it('appends user tags', () => {
    const tags = [{ key: 'malware', value: 'true' }];
    const result = buildUploadFormBase('', ['g1'], tags, []);
    expect('form' in result).toBe(true);
    if ('form' in result) {
      expect(result.form.get('tags[malware]')).toBe('true');
    }
  });

  it('returns error for invalid tags', () => {
    const tags = [{ key: 'INVALID', value: 'bad' }];
    const result = buildUploadFormBase('', ['g1'], tags, []);
    expect('errors' in result).toBe(true);
    if ('errors' in result) {
      expect(result.errors[0]).toMatch(/invalid tags/i);
    }
  });

  it('skips tags with empty key or value', () => {
    const tags = [
      { key: '', value: 'v' },
      { key: 'k', value: '' },
    ];
    const result = buildUploadFormBase('', ['g1'], tags, []);
    expect('form' in result).toBe(true);
    if ('form' in result) {
      expect(result.form.has('tags[]')).toBe(false);
      expect(result.form.has('tags[k]')).toBe(false);
    }
  });
});

describe('appendOriginToForm — Downloaded', () => {
  it('appends origin fields when URL is provided', () => {
    const form = new FormData();
    const origin = originWith({ originType: 'Downloaded', downloaded: { url: 'https://evil.com', name: 'evil' } });
    const result = appendOriginToForm(form, origin);
    expect(result.success).toBe(true);
    expect(form.get('origin[origin_type]')).toBe('Downloaded');
    expect(form.get('origin[url]')).toBe('https://evil.com');
    expect(form.get('origin[name]')).toBe('evil');
  });

  it('succeeds with URL only (no name)', () => {
    const form = new FormData();
    const origin = originWith({ originType: 'Downloaded', downloaded: { url: 'https://x.com', name: '' } });
    const result = appendOriginToForm(form, origin);
    expect(result.success).toBe(true);
    expect(form.get('origin[url]')).toBe('https://x.com');
    expect(form.has('origin[name]')).toBe(false);
  });

  it('returns error when name set but URL blank', () => {
    const form = new FormData();
    const origin = originWith({ originType: 'Downloaded', downloaded: { url: '', name: 'evil' } });
    const result = appendOriginToForm(form, origin);
    expect(result.success).toBe(false);
    if (!result.success) expect(result.error).toMatch(/URL/);
  });

  it('is a no-op when both fields are blank', () => {
    const form = new FormData();
    const origin = originWith({ originType: 'Downloaded', downloaded: { url: '', name: '' } });
    const result = appendOriginToForm(form, origin);
    expect(result.success).toBe(true);
    expect(form.has('origin[origin_type]')).toBe(false);
  });
});

describe('appendOriginToForm — Transformed', () => {
  it('appends fields when parent is provided', () => {
    const form = new FormData();
    const origin = originWith({
      originType: 'Transformed',
      transformed: { parentFile: 'abc'.repeat(21) + 'a', tool: 'upx', toolFlags: '-d' },
    });
    const result = appendOriginToForm(form, origin);
    expect(result.success).toBe(true);
    expect(form.get('origin[origin_type]')).toBe('Transformed');
    expect(form.get('origin[parent]')).toBeTruthy();
    expect(form.get('origin[tool]')).toBe('upx');
    expect(form.get('origin[flags]')).toBe('-d');
  });

  it('returns error when tool set but parent blank', () => {
    const form = new FormData();
    const origin = originWith({
      originType: 'Transformed',
      transformed: { parentFile: '', tool: 'upx', toolFlags: '' },
    });
    const result = appendOriginToForm(form, origin);
    expect(result.success).toBe(false);
    if (!result.success) expect(result.error).toMatch(/TOOL.*PARENT/);
  });

  it('returns error when flags set but parent blank', () => {
    const form = new FormData();
    const origin = originWith({
      originType: 'Transformed',
      transformed: { parentFile: '', tool: '', toolFlags: '-d' },
    });
    const result = appendOriginToForm(form, origin);
    expect(result.success).toBe(false);
    if (!result.success) expect(result.error).toMatch(/FLAGS.*PARENT/);
  });
});

describe('appendOriginToForm — Unpacked', () => {
  it('appends fields when parent is provided', () => {
    const form = new FormData();
    const origin = originWith({
      originType: 'Unpacked',
      unpacked: { parentFile: 'a'.repeat(64), tool: '7z', toolFlags: '' },
    });
    const result = appendOriginToForm(form, origin);
    expect(result.success).toBe(true);
    expect(form.get('origin[origin_type]')).toBe('Unpacked');
    expect(form.get('origin[tool]')).toBe('7z');
  });

  it('returns error when tool set but parent blank', () => {
    const form = new FormData();
    const origin = originWith({
      originType: 'Unpacked',
      unpacked: { parentFile: '', tool: 'unzip', toolFlags: '' },
    });
    const result = appendOriginToForm(form, origin);
    expect(result.success).toBe(false);
  });
});

describe('appendOriginToForm — Carved', () => {
  it('appends fields when parent and type are provided', () => {
    const form = new FormData();
    const origin = originWith({
      originType: 'Carved',
      carved: {
        parentFile: 'a'.repeat(64),
        tool: 'foremost',
        carvedType: 'Pcap',
        pcap: {
          sourceIp: '1.2.3.4',
          destinationIp: '5.6.7.8',
          sourcePort: '80',
          destinationPort: '443',
          protocol: 'TCP',
          url: 'http://x.com',
        },
      },
    });
    const result = appendOriginToForm(form, origin);
    expect(result.success).toBe(true);
    expect(form.get('origin[origin_type]')).toBe('CarvedPcap');
    expect(form.get('origin[src_ip]')).toBe('1.2.3.4');
    expect(form.get('origin[dest_ip]')).toBe('5.6.7.8');
    expect(form.get('origin[src_port]')).toBe('80');
    expect(form.get('origin[dest_port]')).toBe('443');
    expect(form.get('origin[proto]')).toBe('TCP');
    expect(form.get('origin[url]')).toBe('http://x.com');
  });

  it('uses CarvedUnknown type', () => {
    const form = new FormData();
    const origin = originWith({
      originType: 'Carved',
      carved: {
        parentFile: 'b'.repeat(64),
        tool: '',
        carvedType: 'Unknown',
        pcap: { sourceIp: '', destinationIp: '', sourcePort: '', destinationPort: '', protocol: '', url: '' },
      },
    });
    const result = appendOriginToForm(form, origin);
    expect(result.success).toBe(true);
    expect(form.get('origin[origin_type]')).toBe('CarvedUnknown');
  });

  it('returns error when tool set but parent blank', () => {
    const form = new FormData();
    const origin = originWith({
      originType: 'Carved',
      carved: {
        parentFile: '',
        tool: 'foremost',
        carvedType: 'Pcap',
        pcap: { sourceIp: '', destinationIp: '', sourcePort: '', destinationPort: '', protocol: '', url: '' },
      },
    });
    const result = appendOriginToForm(form, origin);
    expect(result.success).toBe(false);
  });

  it('does not append PCAP fields for Unknown type', () => {
    const form = new FormData();
    const origin = originWith({
      originType: 'Carved',
      carved: {
        parentFile: 'c'.repeat(64),
        tool: '',
        carvedType: 'Unknown',
        pcap: { sourceIp: '1.2.3.4', destinationIp: '', sourcePort: '', destinationPort: '', protocol: '', url: '' },
      },
    });
    appendOriginToForm(form, origin);
    expect(form.has('origin[src_ip]')).toBe(false);
  });
});

describe('appendOriginToForm — Wire', () => {
  it('appends fields when sniffer is provided', () => {
    const form = new FormData();
    const origin = originWith({
      originType: 'Wire',
      wire: { sniffer: 'wireshark', source: '10.0.0.1', destination: '10.0.0.2' },
    });
    const result = appendOriginToForm(form, origin);
    expect(result.success).toBe(true);
    expect(form.get('origin[origin_type]')).toBe('Wire');
    expect(form.get('origin[sniffer]')).toBe('wireshark');
    expect(form.get('origin[source]')).toBe('10.0.0.1');
    expect(form.get('origin[destination]')).toBe('10.0.0.2');
  });

  it('returns error when source set but sniffer blank', () => {
    const form = new FormData();
    const origin = originWith({
      originType: 'Wire',
      wire: { sniffer: '', source: '10.0.0.1', destination: '' },
    });
    const result = appendOriginToForm(form, origin);
    expect(result.success).toBe(false);
    if (!result.success) expect(result.error).toMatch(/SOURCE.*SNIFFER/);
  });

  it('returns error when destination set but sniffer blank', () => {
    const form = new FormData();
    const origin = originWith({
      originType: 'Wire',
      wire: { sniffer: '', source: '', destination: '10.0.0.2' },
    });
    const result = appendOriginToForm(form, origin);
    expect(result.success).toBe(false);
    if (!result.success) expect(result.error).toMatch(/DESTINATION.*SNIFFER/);
  });
});

describe('appendOriginToForm — Incident', () => {
  it('appends all fields when incident ID is provided', () => {
    const form = new FormData();
    const origin = originWith({
      originType: 'Incident',
      incident: {
        incident: 'INC-001',
        coverTerm: 'ALPHA',
        missionTeam: 'TEAM-A',
        network: 'DMZ',
        machine: 'SRV-01',
        location: 'DC-EAST',
      },
    });
    const result = appendOriginToForm(form, origin);
    expect(result.success).toBe(true);
    expect(form.get('origin[origin_type]')).toBe('Incident');
    expect(form.get('origin[incident]')).toBe('INC-001');
    expect(form.get('origin[cover_term]')).toBe('ALPHA');
    expect(form.get('origin[mission_team]')).toBe('TEAM-A');
    expect(form.get('origin[network]')).toBe('DMZ');
    expect(form.get('origin[machine]')).toBe('SRV-01');
    expect(form.get('origin[location]')).toBe('DC-EAST');
  });

  const optionalFields = ['coverTerm', 'missionTeam', 'network', 'machine', 'location'] as const;
  for (const field of optionalFields) {
    it(`returns error when ${field} set but incident ID blank`, () => {
      const form = new FormData();
      const incident = { incident: '', coverTerm: '', missionTeam: '', network: '', machine: '', location: '' };
      incident[field] = 'value';
      const origin = originWith({ originType: 'Incident', incident });
      const result = appendOriginToForm(form, origin);
      expect(result.success).toBe(false);
      if (!result.success) expect(result.error).toMatch(/INCIDENT ID/);
    });
  }
});

describe('appendOriginToForm — MemoryDump', () => {
  it('appends all fields when memoryType is provided', () => {
    const form = new FormData();
    const origin = originWith({
      originType: 'MemoryDump',
      memoryDump: {
        memoryType: 'LSASS',
        parentFile: 'a'.repeat(64),
        reconstructed: 'true',
        baseAddress: '0x7FFE0000',
      },
    });
    const result = appendOriginToForm(form, origin);
    expect(result.success).toBe(true);
    expect(form.get('origin[origin_type]')).toBe('MemoryDump');
    expect(form.get('origin[memory_type]')).toBe('LSASS');
    expect(form.get('origin[parent]')).toBe('a'.repeat(64));
    expect(form.get('origin[reconstructed]')).toBe('true');
    expect(form.get('origin[base_addr]')).toBe('0x7FFE0000');
  });

  it('returns error when parentFile set but memoryType blank', () => {
    const form = new FormData();
    const origin = originWith({
      originType: 'MemoryDump',
      memoryDump: { memoryType: '', parentFile: 'a'.repeat(64), reconstructed: '', baseAddress: '' },
    });
    const result = appendOriginToForm(form, origin);
    expect(result.success).toBe(false);
    if (!result.success) expect(result.error).toMatch(/MEMORY TYPE/);
  });

  it('returns error when reconstructed set but memoryType blank', () => {
    const form = new FormData();
    const origin = originWith({
      originType: 'MemoryDump',
      memoryDump: { memoryType: '', parentFile: '', reconstructed: 'true', baseAddress: '' },
    });
    const result = appendOriginToForm(form, origin);
    expect(result.success).toBe(false);
    if (!result.success) expect(result.error).toMatch(/MEMORY TYPE/);
  });

  it('returns error when baseAddress set but memoryType blank', () => {
    const form = new FormData();
    const origin = originWith({
      originType: 'MemoryDump',
      memoryDump: { memoryType: '', parentFile: '', reconstructed: '', baseAddress: '0x1000' },
    });
    const result = appendOriginToForm(form, origin);
    expect(result.success).toBe(false);
    if (!result.success) expect(result.error).toMatch(/MEMORY TYPE/);
  });
});
