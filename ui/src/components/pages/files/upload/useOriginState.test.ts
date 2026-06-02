import { describe, it, expect } from 'vitest';

// project imports
import { originReducer, OriginAction } from './useOriginState';
import { DEFAULT_ORIGIN_STATE, OriginState } from './types';

function stateWith(patch: Partial<OriginState>): OriginState {
  return { ...structuredClone(DEFAULT_ORIGIN_STATE), ...patch };
}

describe('originReducer', () => {
  it('SET_ORIGIN_TYPE changes origin type', () => {
    const next = originReducer(DEFAULT_ORIGIN_STATE, {
      type: 'SET_ORIGIN_TYPE',
      originType: 'Wire',
    });
    expect(next.originType).toBe('Wire');
  });

  it('SET_DOWNLOADED_FIELD updates url', () => {
    const next = originReducer(DEFAULT_ORIGIN_STATE, {
      type: 'SET_DOWNLOADED_FIELD',
      field: 'url',
      value: 'https://evil.com',
    });
    expect(next.downloaded.url).toBe('https://evil.com');
    expect(next.downloaded.name).toBe('');
  });

  it('SET_DOWNLOADED_FIELD updates name', () => {
    const next = originReducer(DEFAULT_ORIGIN_STATE, {
      type: 'SET_DOWNLOADED_FIELD',
      field: 'name',
      value: 'evil.com',
    });
    expect(next.downloaded.name).toBe('evil.com');
  });

  it('SET_PARENT_TOOL_FIELD updates transformed variant', () => {
    const next = originReducer(DEFAULT_ORIGIN_STATE, {
      type: 'SET_PARENT_TOOL_FIELD',
      variant: 'transformed',
      field: 'parentFile',
      value: 'a'.repeat(64),
    });
    expect(next.transformed.parentFile).toBe('a'.repeat(64));
    expect(next.unpacked.parentFile).toBe('');
  });

  it('SET_PARENT_TOOL_FIELD updates unpacked variant', () => {
    const next = originReducer(DEFAULT_ORIGIN_STATE, {
      type: 'SET_PARENT_TOOL_FIELD',
      variant: 'unpacked',
      field: 'tool',
      value: '7z',
    });
    expect(next.unpacked.tool).toBe('7z');
    expect(next.transformed.tool).toBe('');
  });

  it('SET_PARENT_TOOL_FIELD updates toolFlags', () => {
    const next = originReducer(DEFAULT_ORIGIN_STATE, {
      type: 'SET_PARENT_TOOL_FIELD',
      variant: 'transformed',
      field: 'toolFlags',
      value: '-d --best',
    });
    expect(next.transformed.toolFlags).toBe('-d --best');
  });

  it('SET_CARVED_FIELD updates parentFile', () => {
    const next = originReducer(DEFAULT_ORIGIN_STATE, {
      type: 'SET_CARVED_FIELD',
      field: 'parentFile',
      value: 'b'.repeat(64),
    });
    expect(next.carved.parentFile).toBe('b'.repeat(64));
  });

  it('SET_CARVED_FIELD updates tool', () => {
    const next = originReducer(DEFAULT_ORIGIN_STATE, {
      type: 'SET_CARVED_FIELD',
      field: 'tool',
      value: 'foremost',
    });
    expect(next.carved.tool).toBe('foremost');
  });

  it('SET_CARVED_PCAP_FIELD updates nested pcap field', () => {
    const next = originReducer(DEFAULT_ORIGIN_STATE, {
      type: 'SET_CARVED_PCAP_FIELD',
      field: 'sourceIp',
      value: '192.168.1.1',
    });
    expect(next.carved.pcap.sourceIp).toBe('192.168.1.1');
    expect(next.carved.pcap.destinationIp).toBe('');
  });

  it('SET_CARVED_PCAP_FIELD preserves other pcap fields', () => {
    const state = stateWith({
      carved: {
        ...DEFAULT_ORIGIN_STATE.carved,
        pcap: { ...DEFAULT_ORIGIN_STATE.carved.pcap, sourceIp: '1.2.3.4' },
      },
    });
    const next = originReducer(state, {
      type: 'SET_CARVED_PCAP_FIELD',
      field: 'destinationIp',
      value: '5.6.7.8',
    });
    expect(next.carved.pcap.sourceIp).toBe('1.2.3.4');
    expect(next.carved.pcap.destinationIp).toBe('5.6.7.8');
  });

  it('SET_CARVED_TYPE changes carved subtype', () => {
    const next = originReducer(DEFAULT_ORIGIN_STATE, {
      type: 'SET_CARVED_TYPE',
      carvedType: 'Unknown',
    });
    expect(next.carved.carvedType).toBe('Unknown');
  });

  it('SET_WIRE_FIELD updates sniffer', () => {
    const next = originReducer(DEFAULT_ORIGIN_STATE, {
      type: 'SET_WIRE_FIELD',
      field: 'sniffer',
      value: 'wireshark',
    });
    expect(next.wire.sniffer).toBe('wireshark');
  });

  it('SET_WIRE_FIELD updates source and destination', () => {
    let next = originReducer(DEFAULT_ORIGIN_STATE, {
      type: 'SET_WIRE_FIELD',
      field: 'source',
      value: '10.0.0.1',
    });
    next = originReducer(next, {
      type: 'SET_WIRE_FIELD',
      field: 'destination',
      value: '10.0.0.2',
    });
    expect(next.wire.source).toBe('10.0.0.1');
    expect(next.wire.destination).toBe('10.0.0.2');
  });

  it('SET_INCIDENT_FIELD updates each field independently', () => {
    const fields = ['incident', 'coverTerm', 'missionTeam', 'network', 'machine', 'location'] as const;
    let state: OriginState = DEFAULT_ORIGIN_STATE;
    for (const field of fields) {
      state = originReducer(state, {
        type: 'SET_INCIDENT_FIELD',
        field,
        value: `val-${field}`,
      });
    }
    for (const field of fields) {
      expect(state.incident[field]).toBe(`val-${field}`);
    }
  });

  it('SET_MEMORY_DUMP_FIELD updates each field independently', () => {
    const fields = ['memoryType', 'parentFile', 'reconstructed', 'baseAddress'] as const;
    let state: OriginState = DEFAULT_ORIGIN_STATE;
    for (const field of fields) {
      state = originReducer(state, {
        type: 'SET_MEMORY_DUMP_FIELD',
        field,
        value: `val-${field}`,
      });
    }
    for (const field of fields) {
      expect(state.memoryDump[field]).toBe(`val-${field}`);
    }
  });

  it('RESET returns DEFAULT_ORIGIN_STATE', () => {
    const state = stateWith({
      originType: 'Wire',
      wire: { sniffer: 'tcpdump', source: '1.1.1.1', destination: '2.2.2.2' },
    });
    const next = originReducer(state, { type: 'RESET' });
    expect(next).toEqual(DEFAULT_ORIGIN_STATE);
  });

  it('returns state unchanged for unknown action', () => {
    const state = structuredClone(DEFAULT_ORIGIN_STATE);
    // @ts-expect-error testing unknown action
    const next = originReducer(state, { type: 'UNKNOWN_ACTION' } as OriginAction);
    expect(next).toEqual(state);
  });

  it('does not mutate previous state', () => {
    const state = structuredClone(DEFAULT_ORIGIN_STATE);
    const frozen = JSON.stringify(state);
    originReducer(state, {
      type: 'SET_DOWNLOADED_FIELD',
      field: 'url',
      value: 'https://test.com',
    });
    expect(JSON.stringify(state)).toBe(frozen);
  });
});
