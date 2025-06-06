import { Tags } from './tags';

export type PcapNetworkProtocol = {
  /// The TCP protocol
  /// #[serde(rename = "TCP", alias = "Tcp", alias = "tcp")]
  Tcp?: 'TCP' | 'Tcp' | 'tcp';
  /// The UDP protocol
  /// #[serde(rename = "UDP", alias = "Udp", alias = "udp")]
  Udp?: 'UDP' | 'Udp' | 'udp';
};

export type CarvedOrigin = {
  /// The sample was carved from a packet capture
  Pcap?: {
    /// The source IP this sample was sent from
    /// #[cfg_attr(feature = "api", schema(value_type = String))]
    src_ip?: string;
    /// The destination IP this sample was going to
    /// #[cfg_attr(feature = "api", schema(value_type = String))]
    dest_ip?: string;
    /// The source port this sample was sent from
    src_port?: number;
    /// The destination port this sample was going to
    dest_port?: number;
    /// The type of protocol this sample was transported in
    proto?: PcapNetworkProtocol;
    /// The URL this file was retrieved from or sent to
    url?: string;
  };
  /// The sample was carved from an unknown or unspecified file type
  Unknown?: 'Unknown';
};

export type Origin = {
  /// This sample was downloaded from an external website
  Downloaded?: { url: string; name?: string };
  /// This sample was unpacked from another sample
  Unpacked?: {
    /// The tool that unpacked this sample
    tool?: string;
    /// The sha256 of the sample this was unpacked from
    parent: string;
    /// Whether this parent sample exists or not
    dangling: boolean;
  };
  /// This sample is an output of a transformation of another sample
  Transformed?: {
    /// The tool that transformed this sample
    tool?: string;
    /// The sha256 of the sample that was transformed
    parent: string;
    /// Whether this parent sample exists or not
    dangling: boolean;
    /// The flags used to transform this sample
    flags: string[];
    /// The full command used to transform this sample
    cmd?: string;
  };
  /// This sample comes from a sniffer/the wire
  Wire?: {
    /// The name of the sniffer that found this sample
    sniffer: string;
    /// The source of this sample on the wire
    source?: string;
    /// The destination of this sample on the wire
    destination?: string;
  };
  /// This sample comes from an incident or engagement
  Incident?: {
    /// The incident this sample comes from
    incident: string;
    /// The cover term used for this incident
    cover_term?: string;
    /// The mission team involved in this incident
    mission_team?: string;
    /// The network this sample was found on
    network?: string;
    /// The machine this sample was found on
    machine?: string;
    /// The location this sample is from
    location?: string;
  };
  /// This sample comes from dumping memory while running a parent sample
  MemoryDump?: {
    /// The sample this file was a memory dump from
    parent: string;
    /// Whether this parent sample exists or not
    dangling: boolean;
    /// The characteristics that were reconstructed in this memory dump
    reconstructed: string[];
    /// the base address for this memory dump
    base_addr?: string;
  };
  /// This sample was built from source
  Source?: {
    /// The repository this was built from
    repo: string;
    /// The branch, commit, tag that this was built from
    commitish?: string;
    /// The commit the repository was on
    commit: string;
    /// The flags used to build this
    flags: string[];
    /// The build system that was used to build this
    system: string;
    /// Whether this is a supporting build file or a final build file
    supporting: boolean;
  };
  /// This sample was statically carved out from another sample
  ///
  /// Unlike `Unpacked`, `Carved` describes a sample that is just
  /// a simple piece of another file, like a file from an archive or
  /// a packet capture. It's extraction can be easily replicated without
  /// any dynamic unpacking process.
  Carved?: {
    /// The sample this file was carved from
    parent: string;
    /// The tool that carved out this sample
    tool?: string;
    /// Whether this parent sample exists or not
    dangling: boolean;
    /// The type of carved file this is
    carved_origin: CarvedOrigin;
  };
  /// This sample has no unique origin
  None?: 'None';
};

export type SubmissionChunk = {
  /// A UUID for this submission
  id: string;
  /// The name of this sample if one was specified
  name?: string;
  /// A description for this sample
  description: String;
  /// The groups this submission is in
  groups: string[];
  /// The user who submitted this sample
  submitter: string;
  /// When this sample was uploaded
  uploaded: string;
  /// The origin of this sample if one was specified
  origin: Origin;
};

export type Sample = {
  /// The sha256 of this sample
  sha256: string;
  /// The sha1 of this sample
  sha1: string;
  /// The md5 of this sample
  md5: string;
  /// The tags for this sample
  tags: Tags;
  /// The different submissions for this sample
  submissions: SubmissionChunk[];
  /// Any comments for this sample
  comments: Comment[];
};
