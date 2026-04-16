import { useReducer } from 'react';
import { CarvedSubType, OriginState, OriginType, DEFAULT_ORIGIN_STATE } from './types';

type OriginAction =
  | { type: 'SET_ORIGIN_TYPE'; originType: OriginType }
  | { type: 'SET_CARVED_TYPE'; carvedType: CarvedSubType }
  | { type: 'SET_DOWNLOADED_FIELD'; field: 'url' | 'name'; value: string }
  | { type: 'SET_PARENT_TOOL_FIELD'; variant: 'transformed' | 'unpacked'; field: 'parentFile' | 'tool' | 'toolFlags'; value: string }
  | { type: 'SET_CARVED_FIELD'; field: 'parentFile' | 'tool'; value: string }
  | {
      type: 'SET_CARVED_PCAP_FIELD';
      field: 'sourceIp' | 'destinationIp' | 'sourcePort' | 'destinationPort' | 'protocol' | 'url';
      value: string;
    }
  | { type: 'SET_WIRE_FIELD'; field: 'sniffer' | 'source' | 'destination'; value: string }
  | { type: 'SET_INCIDENT_FIELD'; field: 'incident' | 'coverTerm' | 'missionTeam' | 'network' | 'machine' | 'location'; value: string }
  | { type: 'SET_MEMORY_DUMP_FIELD'; field: 'memoryType' | 'parentFile' | 'reconstructed' | 'baseAddress'; value: string }
  | { type: 'RESET' };

function originReducer(state: OriginState, action: OriginAction): OriginState {
  switch (action.type) {
    case 'SET_ORIGIN_TYPE':
      return { ...state, originType: action.originType };
    case 'SET_CARVED_TYPE':
      return { ...state, carved: { ...state.carved, carvedType: action.carvedType } };
    case 'SET_DOWNLOADED_FIELD':
      return { ...state, downloaded: { ...state.downloaded, [action.field]: action.value } };
    case 'SET_PARENT_TOOL_FIELD':
      return { ...state, [action.variant]: { ...state[action.variant], [action.field]: action.value } };
    case 'SET_CARVED_FIELD':
      return { ...state, carved: { ...state.carved, [action.field]: action.value } };
    case 'SET_CARVED_PCAP_FIELD':
      return { ...state, carved: { ...state.carved, pcap: { ...state.carved.pcap, [action.field]: action.value } } };
    case 'SET_WIRE_FIELD':
      return { ...state, wire: { ...state.wire, [action.field]: action.value } };
    case 'SET_INCIDENT_FIELD':
      return { ...state, incident: { ...state.incident, [action.field]: action.value } };
    case 'SET_MEMORY_DUMP_FIELD':
      return { ...state, memoryDump: { ...state.memoryDump, [action.field]: action.value } };
    case 'RESET':
      return DEFAULT_ORIGIN_STATE;
    default:
      return state;
  }
}

export function useOriginState() {
  const [originState, dispatch] = useReducer(originReducer, DEFAULT_ORIGIN_STATE);

  const setDownloadedField = (field: 'url' | 'name', value: string) => {
    dispatch({ type: 'SET_DOWNLOADED_FIELD', field, value });
  };

  const setParentToolField = (variant: 'transformed' | 'unpacked', field: 'parentFile' | 'tool' | 'toolFlags', value: string) => {
    dispatch({ type: 'SET_PARENT_TOOL_FIELD', variant, field, value });
  };

  const setCarvedField = (field: 'parentFile' | 'tool', value: string) => {
    dispatch({ type: 'SET_CARVED_FIELD', field, value });
  };

  const setCarvedPcapField = (
    field: 'sourceIp' | 'destinationIp' | 'sourcePort' | 'destinationPort' | 'protocol' | 'url',
    value: string,
  ) => {
    dispatch({ type: 'SET_CARVED_PCAP_FIELD', field, value });
  };

  const setCarvedType = (carvedType: CarvedSubType) => {
    dispatch({ type: 'SET_CARVED_TYPE', carvedType });
  };

  const setWireField = (field: 'sniffer' | 'source' | 'destination', value: string) => {
    dispatch({ type: 'SET_WIRE_FIELD', field, value });
  };

  const setIncidentField = (field: 'incident' | 'coverTerm' | 'missionTeam' | 'network' | 'machine' | 'location', value: string) => {
    dispatch({ type: 'SET_INCIDENT_FIELD', field, value });
  };

  const setMemoryDumpField = (field: 'memoryType' | 'parentFile' | 'reconstructed' | 'baseAddress', value: string) => {
    dispatch({ type: 'SET_MEMORY_DUMP_FIELD', field, value });
  };

  const setOriginType = (originType: OriginType) => {
    dispatch({ type: 'SET_ORIGIN_TYPE', originType });
  };

  const resetOriginState = () => {
    dispatch({ type: 'RESET' });
  };

  return {
    originState,
    setOriginType,
    setDownloadedField,
    setParentToolField,
    setCarvedField,
    setCarvedPcapField,
    setCarvedType,
    setWireField,
    setIncidentField,
    setMemoryDumpField,
    resetOriginState,
  };
}
