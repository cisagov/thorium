import { CreateEntity, Entities, Entity } from './entities';

export enum TransportLayerProtocol {
  /// A protocol for reliable ordered delivery of data
  TCP = 'TCP',
  /// A lightweight protocol without error correction or acknowledgement
  UDP = 'UDP',
  /// A reliable message oriented protocol supporting multi-streaming and multi-homing
  SCTP = 'SCTP',
}

export enum NetConState {
  /// This connection is waiting for a external connection request
  Listen = 'Listen',
  /// This connection has sent a connection request and is waiting for a response
  Syn = 'Syn',
  /// This connection has sent a Syn packet and received a sync-ack
  SynAck = 'SynAck',
  /// This connection is open and can send and receive data
  Established = 'Established',
  /// One side has sent a closure request and is waiting for an acknowledgement
  Fin = 'Fin',
  /// The destination side has requested to terminate this connection
  CloseWait = 'CloseWait',
  /// The destination side is waiting for the final acknowledgement after sending its own fin packet
  LastAck = 'LastAck',
  /// This connection is waiting for the destination to receive a connection termination request
  TimeWait = 'TimeWait',
  /// This connection has closed
  Closed = 'Closed',
}

export type NetworkConnectionMetaFields = {
  /// The protocol this network connection is using at the transport layer
  protocol?: TransportLayerProtocol;
  /// The source address for this connection
  source: string;
  /// source port for this connection
  source_port?: number;
  /// The destination address for this connection
  destination?: string;
  /// The destination port for this connection
  destination_port: number;
  /// The state of this connection
  state?: NetConState;
  /// The pid this process is from
  pid?: BigInt;
  /// The name of this process that owns this connection
  process?: string;
  /// When this process was created (not in Thorium but the actually network connection)
  create_time?: string;
};

export type NetworkConnectionCreateMetaFields = NetworkConnectionMetaFields;

export type NetworkConnectionMeta = {
  NetworkConnection: NetworkConnectionMetaFields;
};

export type NetworkConnectionCreateMeta = {
  NetworkConnection: NetworkConnectionCreateMetaFields;
};

export type NetworkConnection = Entity<Entities.NetworkConnection>;

export type CreateNetworkConnection = CreateEntity<Entities.NetworkConnection>;

export const BlankNetworkConnection: NetworkConnection = {
  id: '',
  name: '',
  groups: [],
  description: null,
  kind: Entities.NetworkConnection,
  metadata: {
    NetworkConnection: {
      source: '0.0.0.0',
      destination_port: 80,
    },
  },
  tags: {},
  submitter: '',
  created: '',
};

export const BlankCreateNetworkConnection: CreateNetworkConnection = {
  name: '',
  groups: [],
  tags: {},
  description: null,
  kind: Entities.NetworkConnection,
  metadata: {
    NetworkConnection: {
      source: '0.0.0.0',
      destination_port: 80,
    },
  },
};
