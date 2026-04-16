export const REQUIRED_FIELDS = ['title', 'logsource', 'detection'] as const;

export const DETECTION_REQUIRED_FIELDS = ['condition'] as const;

export const STATUS_VALUES = ['stable', 'test', 'experimental', 'deprecated', 'unsupported'] as const;

export const LEVEL_VALUES = ['informational', 'low', 'medium', 'high', 'critical'] as const;

export const RELATED_TYPES = ['derived', 'obsolete', 'merged', 'renamed', 'similar'] as const;

export const LOGSOURCE_FIELDS = ['category', 'product', 'service', 'definition'] as const;

export const TITLE_MAX_LENGTH = 256;

export const NAME_MAX_LENGTH = 256;

export const TAXONOMY_MAX_LENGTH = 256;

export const DESCRIPTION_MAX_LENGTH = 65535;

export const FALSEPOSITIVES_MIN_LENGTH = 2;

export const SCOPE_MIN_LENGTH = 2;

export const UUID_V4_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export const DATE_PATTERN = /^\d{4}-(0[1-9]|1[012])-(0[1-9]|[12]\d|3[01])$/;

export const TAG_PATTERN = /^[a-z0-9_-]+\.[a-z0-9._-]+$/;

export const KNOWN_TOP_LEVEL_FIELDS = [
  'title',
  'id',
  'name',
  'related',
  'taxonomy',
  'status',
  'description',
  'license',
  'author',
  'references',
  'date',
  'modified',
  'logsource',
  'detection',
  'fields',
  'falsepositives',
  'level',
  'tags',
  'scope',
] as const;

export const COMMON_LOGSOURCE_CATEGORIES = [
  'process_creation',
  'file_event',
  'file_change',
  'file_rename',
  'file_delete',
  'file_access',
  'image_load',
  'network_connection',
  'registry_set',
  'registry_add',
  'registry_delete',
  'registry_event',
  'registry_rename',
  'dns_query',
  'firewall',
  'webserver',
  'proxy',
  'antivirus',
  'create_remote_thread',
  'create_stream_hash',
  'pipe_created',
  'ps_classic_start',
  'ps_module',
  'ps_script',
  'sysmon_error',
  'sysmon_status',
  'wmi_event',
  'driver_load',
  'process_access',
  'process_tampering',
  'process_termination',
  'raw_access_thread',
  'clipboard_capture',
] as const;

export const COMMON_LOGSOURCE_PRODUCTS = [
  'windows',
  'linux',
  'macos',
  'azure',
  'aws',
  'gcp',
  'github',
  'okta',
  'm365',
  'onelogin',
  'qualys',
  'cisco',
  'django',
  'spring',
  'sql',
  'apache',
  'nginx',
  'zeek',
] as const;

export const COMMON_LOGSOURCE_SERVICES = [
  'security',
  'system',
  'application',
  'sshd',
  'syslog',
  'auditd',
  'powershell',
  'powershell-classic',
  'dns-server',
  'firewall-as',
  'applocker',
  'windefend',
  'bits-client',
  'codeintegrity-operational',
  'msexchange-management',
  'printservice-admin',
  'printservice-operational',
  'smbclient-security',
  'taskscheduler',
  'terminalservices-localsessionmanager',
  'wmi',
  'driver-framework',
  'ntlm',
  'ldap_debug',
  'diagnosis-scripted',
  'shell-core',
  'openssh',
] as const;

export const VALUE_MODIFIERS = [
  'all',
  'startswith',
  'endswith',
  'contains',
  'exists',
  'cased',
  'neq',
  'windash',
  're',
  'i',
  'm',
  's',
  'base64',
  'base64offset',
  'utf16le',
  'utf16be',
  'utf16',
  'wide',
  'lt',
  'lte',
  'gt',
  'gte',
  'cidr',
  'expand',
  'fieldref',
  'minute',
  'hour',
  'day',
  'week',
  'month',
  'year',
] as const;
