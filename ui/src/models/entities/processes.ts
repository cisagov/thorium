import { CreateEntity, Entities, Entity } from './entities';

// ----- Windows -----

export type WindowsProcessMetaFields = {
  /// This processes id
  pid: number;
  /// This processes parent PID
  parent_pid?: bigint;
  /// The name of the executable for this processes
  name?: string;
  /// The path to this executable
  image_path?: string;
  /// The full cmd for this process
  command?: string;
  /// The offset for this process
  offset?: bigint;
  /// the number of threads this process spawned
  threads?: number;
  /// The number of handles this process had open
  handles?: number;
  /// Whether this process is using the wow64 emulator or not
  is_wow64?: boolean;
  /// The session id for this process
  session_id?: number;
  /// When this process was spawned (not created in Thorium)
  create_time: string;
  /// When this process exited
  exit_time?: string;
};

export type WindowsProcessCreateMetaFields = WindowsProcessMetaFields;

export type WindowsProcessMeta = {
  WindowsProcess: WindowsProcessMetaFields;
};

export type WindowsProcessCreateMeta = {
  WindowsProcess: WindowsProcessCreateMetaFields;
};

export type WindowsProcess = Entity<Entities.WindowsProcess>;

export type CreateWindowsProcess = CreateEntity<Entities.WindowsProcess>;

export const BlankWindowsProcess: WindowsProcess = {
  id: '',
  name: '',
  groups: [],
  description: null,
  kind: Entities.WindowsProcess,
  metadata: {
    WindowsProcess: {
      pid: 0,
      create_time: '',
    },
  },
  tags: {},
  submitter: '',
  created: '',
};

export const BlankCreateWindowsProcess: CreateWindowsProcess = {
  name: '',
  groups: [],
  tags: {},
  description: null,
  kind: Entities.WindowsProcess,
  metadata: {
    WindowsProcess: {
      pid: 0,
      create_time: '',
    },
  },
};

// ----- Linux -----
