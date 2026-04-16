import { OriginState, TagEntry } from './types';
import { hasInvalidTags } from '@utilities/tags';

type ValidationResult = { success: true } | { success: false; error: string };

export function buildUploadFormBase(
  description: string,
  selectedGroups: string[],
  tags: TagEntry[],
  tlpTags: TagEntry[],
): { form: FormData } | { errors: string[] } {
  const form = new FormData();

  if (description) {
    form.append('description', description);
  }

  if (selectedGroups.length === 0) {
    return { errors: ['At least one group must be selected to submit a file'] };
  }
  selectedGroups.forEach((group) => form.append('groups', group));

  if (tlpTags) {
    tlpTags.forEach((tag) => {
      if (tag.key && tag.value) {
        form.append(`tags[${tag.key}]`, tag.value);
      }
    });
  }

  if (tags) {
    if (hasInvalidTags(tags, true)) {
      return { errors: ['Invalid tags present. Please delete or fix before uploading'] };
    }
    tags.forEach((tag) => {
      if (tag.key && tag.value) {
        form.append(`tags[${tag.key}]`, tag.value);
      }
    });
  }

  return { form };
}

export function appendOriginToForm(formBase: FormData, origin: OriginState): ValidationResult {
  const { originType } = origin;

  if (originType === 'Downloaded') {
    return appendDownloadedOrigin(formBase, origin);
  } else if (originType === 'Transformed') {
    return appendTransformedOrigin(formBase, origin);
  } else if (originType === 'Unpacked') {
    return appendUnpackedOrigin(formBase, origin);
  } else if (originType === 'Carved') {
    return appendCarvedOrigin(formBase, origin);
  } else if (originType === 'Wire') {
    return appendWireOrigin(formBase, origin);
  } else if (originType === 'Incident') {
    return appendIncidentOrigin(formBase, origin);
  } else if (originType === 'MemoryDump') {
    return appendMemoryDumpOrigin(formBase, origin);
  }

  return { success: true };
}

function appendDownloadedOrigin(form: FormData, origin: OriginState): ValidationResult {
  const { url, name } = origin.downloaded;
  if (url) {
    form.append('origin[origin_type]', 'Downloaded');
    form.append('origin[url]', url);
    if (name) {
      form.append('origin[name]', name);
    }
  } else if (name) {
    return { success: false, error: 'ORIGIN field "SITE NAME" set while necessary field "URL" is blank' };
  }
  return { success: true };
}

function appendTransformedOrigin(form: FormData, origin: OriginState): ValidationResult {
  const { parentFile, tool, toolFlags } = origin.transformed;
  if (parentFile) {
    form.append('origin[origin_type]', 'Transformed');
    form.append('origin[parent]', parentFile);
    if (tool) {
      form.append('origin[tool]', tool);
    }
    if (toolFlags) {
      form.append('origin[flags]', toolFlags);
    }
  } else if (tool) {
    return { success: false, error: 'ORIGIN field "TOOL" set while necessary field "PARENT" is blank' };
  } else if (toolFlags) {
    return { success: false, error: 'ORIGIN field "FLAGS" set while necessary field "PARENT" is blank' };
  }
  return { success: true };
}

function appendUnpackedOrigin(form: FormData, origin: OriginState): ValidationResult {
  const { parentFile, tool, toolFlags } = origin.unpacked;
  if (parentFile) {
    form.append('origin[origin_type]', 'Unpacked');
    form.append('origin[parent]', parentFile);
    if (tool) {
      form.append('origin[tool]', tool);
    }
    if (toolFlags) {
      form.append('origin[flags]', toolFlags);
    }
  } else if (tool) {
    return { success: false, error: 'ORIGIN field "TOOL" set while necessary field "PARENT" is blank' };
  } else if (toolFlags) {
    return { success: false, error: 'ORIGIN field "FLAGS" set while necessary field "PARENT" is blank' };
  }
  return { success: true };
}

function appendCarvedOrigin(form: FormData, origin: OriginState): ValidationResult {
  const { parentFile, tool, carvedType, pcap } = origin.carved;
  if (parentFile) {
    if (!carvedType) {
      return { success: false, error: 'ORIGIN "Carved" needs a specified type' };
    }
    form.append('origin[parent]', parentFile);
    if (tool) {
      form.append('origin[tool]', tool);
    }
    const totalType = 'Carved' + carvedType;
    form.append('origin[origin_type]', totalType);
    if (totalType === 'CarvedPcap') {
      if (pcap.sourceIp) {
        form.append('origin[src_ip]', pcap.sourceIp);
      }
      if (pcap.destinationIp) {
        form.append('origin[dest_ip]', pcap.destinationIp);
      }
      if (pcap.sourcePort) {
        form.append('origin[src_port]', pcap.sourcePort);
      }
      if (pcap.destinationPort) {
        form.append('origin[dest_port]', pcap.destinationPort);
      }
      if (pcap.protocol) {
        form.append('origin[proto]', pcap.protocol);
      }
      if (pcap.url) {
        form.append('origin[url]', pcap.url);
      }
    }
  } else if (tool) {
    return { success: false, error: 'ORIGIN field "TOOL" set while necessary field "PARENT" is blank' };
  }
  return { success: true };
}

function appendWireOrigin(form: FormData, origin: OriginState): ValidationResult {
  const { sniffer, source, destination } = origin.wire;
  if (sniffer) {
    form.append('origin[origin_type]', 'Wire');
    form.append('origin[sniffer]', sniffer);
    if (source) {
      form.append('origin[source]', source);
    }
    if (destination) {
      form.append('origin[destination]', destination);
    }
  } else if (source) {
    return { success: false, error: 'ORIGIN field "SOURCE" set while necessary field "SNIFFER" is blank' };
  } else if (destination) {
    return { success: false, error: 'ORIGIN field "DESTINATION" set while necessary field "SNIFFER" is blank' };
  }
  return { success: true };
}

function appendIncidentOrigin(form: FormData, origin: OriginState): ValidationResult {
  const { incident, missionTeam, coverTerm, network, machine, location } = origin.incident;
  if (incident) {
    form.append('origin[origin_type]', 'Incident');
    form.append('origin[incident]', incident);
    if (missionTeam) {
      form.append('origin[mission_team]', missionTeam);
    }
    if (coverTerm) {
      form.append('origin[cover_term]', coverTerm);
    }
    if (network) {
      form.append('origin[network]', network);
    }
    if (machine) {
      form.append('origin[machine]', machine);
    }
    if (location) {
      form.append('origin[location]', location);
    }
  } else if (coverTerm) {
    return { success: false, error: 'ORIGIN field "COVER TERM" set while necessary field "INCIDENT ID" is blank' };
  } else if (missionTeam) {
    return { success: false, error: 'ORIGIN field "MISSION TEAM" set while necessary field "INCIDENT ID" is blank' };
  } else if (network) {
    return { success: false, error: 'ORIGIN field "NETWORK" set while necessary field "INCIDENT ID" is blank' };
  } else if (machine) {
    return { success: false, error: 'ORIGIN field "MACHINE" set while necessary field "INCIDENT ID" is blank' };
  } else if (location) {
    return { success: false, error: 'ORIGIN field "LOCATION" set while necessary field "INCIDENT ID" is blank' };
  }
  return { success: true };
}

function appendMemoryDumpOrigin(form: FormData, origin: OriginState): ValidationResult {
  const { memoryType, parentFile, reconstructed, baseAddress } = origin.memoryDump;
  if (memoryType) {
    form.append('origin[origin_type]', 'MemoryDump');
    form.append('origin[memory_type]', memoryType);
    if (parentFile) {
      form.append('origin[parent]', parentFile);
    }
    if (reconstructed) {
      form.append('origin[reconstructed]', reconstructed);
    }
    if (baseAddress) {
      form.append('origin[base_addr]', baseAddress);
    }
  } else if (parentFile) {
    return { success: false, error: 'ORIGIN field "PARENT" set while necessary field "MEMORY TYPE" is blank' };
  } else if (reconstructed.length > 0) {
    return { success: false, error: 'ORIGIN field "RECONSTRUCTED" set while necessary field "MEMORY TYPE" is blank' };
  } else if (baseAddress) {
    return { success: false, error: 'ORIGIN field "BASE ADDRESS" set while necessary field "MEMORY TYPE" is blank' };
  }
  return { success: true };
}
