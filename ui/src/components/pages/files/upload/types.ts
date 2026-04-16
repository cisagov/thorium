import { AssociationCreate, AssociationKind, AssociationTarget, BlankAssociationCreate } from '@models/associations';
import { EntityTypes } from '@models/entities/entities';
import { TagEntry } from '@models/tags';

export { AssociationKind, BlankAssociationCreate };
export type { AssociationCreate, AssociationTarget, TagEntry };

export const PARALLEL_UPLOAD_LIMIT = 5;

export const TLP_COLORS: Record<string, string> = {
  CLEAR: 'tlp-clear',
  GREEN: 'tlp-green',
  AMBER: 'tlp-amber',
  'AMBER+STRICT': 'tlp-amber',
  RED: 'tlp-red',
};

export type TLPSelection = Record<string, boolean>;

export const DEFAULT_TLP_SELECTION: TLPSelection = Object.fromEntries(Object.keys(TLP_COLORS).map((key) => [key, false]));

export type OriginType = 'Downloaded' | 'Transformed' | 'Unpacked' | 'Carved' | 'Wire' | 'Incident' | 'MemoryDump';
export type CarvedSubType = 'Pcap' | 'Unknown';

export interface DownloadedOrigin {
  url: string;
  name: string;
}

export interface ParentToolOrigin {
  parentFile: string;
  tool: string;
  toolFlags: string;
}

export interface PcapFields {
  sourceIp: string;
  destinationIp: string;
  sourcePort: string;
  destinationPort: string;
  protocol: string;
  url: string;
}

export interface CarvedOrigin {
  parentFile: string;
  tool: string;
  carvedType: CarvedSubType;
  pcap: PcapFields;
}

export interface WireOrigin {
  sniffer: string;
  source: string;
  destination: string;
}

export interface IncidentOrigin {
  incident: string;
  coverTerm: string;
  missionTeam: string;
  network: string;
  machine: string;
  location: string;
}

export interface MemoryDumpOrigin {
  memoryType: string;
  parentFile: string;
  reconstructed: string;
  baseAddress: string;
}

export interface OriginState {
  originType: OriginType;
  downloaded: DownloadedOrigin;
  transformed: ParentToolOrigin;
  unpacked: ParentToolOrigin;
  carved: CarvedOrigin;
  wire: WireOrigin;
  incident: IncidentOrigin;
  memoryDump: MemoryDumpOrigin;
}

const DEFAULT_PCAP_FIELDS: PcapFields = {
  sourceIp: '',
  destinationIp: '',
  sourcePort: '',
  destinationPort: '',
  protocol: '',
  url: '',
};

export const DEFAULT_ORIGIN_STATE: OriginState = {
  originType: 'Downloaded',
  downloaded: { url: '', name: '' },
  transformed: { parentFile: '', tool: '', toolFlags: '' },
  unpacked: { parentFile: '', tool: '', toolFlags: '' },
  carved: { parentFile: '', tool: '', carvedType: 'Pcap', pcap: { ...DEFAULT_PCAP_FIELDS } },
  wire: { sniffer: '', source: '', destination: '' },
  incident: { incident: '', coverTerm: '', missionTeam: '', network: '', machine: '', location: '' },
  memoryDump: { memoryType: '', parentFile: '', reconstructed: '', baseAddress: '' },
};

export interface FileUploadStatus {
  progress: number;
  size: number;
  type: 'info' | 'success' | 'warning' | 'danger';
  msg: string;
  sha256: string;
  fileFail: boolean;
  reactionFail: boolean;
}

export interface ReactionSubmitResult {
  id?: string;
  error: string;
  group: string;
  pipeline: string;
  path?: string;
  size?: number;
  sha256?: string;
}

export interface ReactionResultEntry {
  id: string;
  sha256: string;
  result: ReactionSubmitResult;
  submission: FileSubmission;
}

export interface FileSubmission {
  path: string;
  size: number;
}

export interface DropzoneFile extends File {
  readonly path: string;
}

export interface UploadProps {
  entity: EntityTypes | undefined;
}

export interface UploadState {
  uploadError: string[];
  runReactionsRes: ReactionSubmitResult[];
  uploadSHA256: string[];
  uploadInProgress: boolean;
  activeUploads: string[];
  uploadStatus: Record<string, FileUploadStatus>;
  uploadFailures: Record<string, FormData>;
  uploadStatusDropdown: Record<string, boolean>;
  uploadReactionRes: ReactionResultEntry[];
  uploadReactions: Record<string, ReactionSubmitResult[]>;
  uploadReactionFailures: number;
  totalUploadSize: number;
  showUploadStatus: boolean;
}

export const DEFAULT_UPLOAD_STATE: UploadState = {
  uploadError: [],
  runReactionsRes: [],
  uploadSHA256: [],
  uploadInProgress: false,
  activeUploads: [],
  uploadStatus: {},
  uploadFailures: {},
  uploadStatusDropdown: {},
  uploadReactionRes: [],
  uploadReactions: {},
  uploadReactionFailures: 0,
  totalUploadSize: 0,
  showUploadStatus: false,
};

export type UploadAction =
  | { type: 'RESET_STATUS' }
  | { type: 'SET_UPLOAD_ERROR'; errors: string[] }
  | { type: 'APPEND_UPLOAD_ERRORS'; errors: string[] }
  | { type: 'SET_UPLOAD_IN_PROGRESS'; value: boolean }
  | { type: 'SET_SHOW_UPLOAD_STATUS'; value: boolean }
  | { type: 'SET_UPLOAD_STATUS'; status: Record<string, FileUploadStatus> }
  | { type: 'UPDATE_FILE_STATUS'; filePath: string; status: FileUploadStatus }
  | { type: 'ADD_ACTIVE_UPLOAD'; filePath: string }
  | { type: 'REMOVE_ACTIVE_UPLOAD'; filePath: string }
  | { type: 'SET_UPLOAD_STATUS_DROPDOWN'; dropdown: Record<string, boolean> }
  | { type: 'TOGGLE_STATUS_DROPDOWN'; filePath: string }
  | { type: 'ADD_UPLOAD_FAILURE'; filePath: string; form: FormData }
  | { type: 'REMOVE_UPLOAD_FAILURE'; filePath: string }
  | { type: 'SET_UPLOAD_SHA256'; sha256s: string[] }
  | { type: 'SET_UPLOAD_REACTIONS'; reactions: Record<string, ReactionSubmitResult[]> }
  | { type: 'UPDATE_UPLOAD_REACTIONS'; filePath: string; results: ReactionSubmitResult[] }
  | { type: 'UPSERT_REACTION_RESULT'; entry: ReactionResultEntry }
  | { type: 'ADJUST_REACTION_FAILURES'; delta: number }
  | { type: 'SET_TOTAL_UPLOAD_SIZE'; size: number }
  | { type: 'SET_RUN_REACTIONS_RES'; results: ReactionSubmitResult[] }
  | {
      type: 'INIT_UPLOAD';
      filesUploadProgress: Record<string, FileUploadStatus>;
      statusDropdown: Record<string, boolean>;
      uploadSize: number;
      initReactions: Record<string, ReactionSubmitResult[]>;
    };
