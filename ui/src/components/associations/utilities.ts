import { BranchNode, Direction, FilterTags, Graph, SubmissionChunk } from '@models';

// get the file name from full path
export const stripFilePath = (filePath: string): string => {
  // if filename is undefined return full file path
  const fileName = filePath.split('/').pop();
  // handle if pop() returns undefined
  return fileName ? fileName : filePath;
};

// format file submission names for use as graph node labels
export const formatSubmissionNames = (submissions: SubmissionChunk[]) => {
  // Samples and Repos will always have submissions, this is an error we are handling
  if (submissions.length == 0) {
    return 'No Valid Name';
  }
  // get all submission names, if they don't exist the list will contain undefined
  const submissionNames = submissions.map((submission) => stripFilePath(submission.name ? submission.name : ''));
  const names = submissionNames.filter((submission) => submission != undefined);
  const uniqueNames = Array.from(new Set(names));
  // filter out undefined values and join to a usable string name that is comma/space separated
  return uniqueNames.join(', ');
};

// format tag label name
export const formatTagNames = (tags: FilterTags, truncate: boolean) => {
  const tagStrings: string[] = [];
  Object.keys(tags).map((key: string) => {
    tags[key].map((value: string) => {
      tagStrings.push(`"${key}: ${value}"`);
    });
  });
  let tagName = tagStrings.join(', ');
  if (truncate && tagName.length > 30) {
    tagName = tagName.substring(0, 15) + '...' + tagName.substring(tagName.length - 15);
  }
  return tagName;
};

// truncate string based on specified max length
export const truncateString = (base: string, length: number): string => {
  if (base.length > length) {
    if (length <= 15) {
      return base.substring(0, length);
    } else {
      let basePartLength = Math.floor((length - 3) / 2);
      let newString = base.substring(0, basePartLength) + '...';
      if (basePartLength % 2) {
        basePartLength += 1;
      }
      return newString + base.substring(base.length - basePartLength);
    }
  }
  return base;
};

// get node name from data_map node data
export const getNodeName = (nodeData: any, maxLength: number): string => {
  if ('Sample' in nodeData) {
    return truncateString(formatSubmissionNames(nodeData.Sample.submissions), maxLength);
  } else if ('Repo' in nodeData) {
    return truncateString(nodeData.Repo.url, maxLength);
  } else if ('Tag' in nodeData) {
    return truncateString(formatTagNames(nodeData.Tag.tags, true), maxLength);
  } else if ('Entity' in nodeData) {
    return truncateString(nodeData.Entity.name, maxLength);
  }
  return '';
};

// build edge label string from branch node structure
export const getEdgeLabel = (target: string, source: string, node: BranchNode, graph: Graph): string => {
  if (node.relationship.Origin) {
    const origin = node.relationship.Origin;
    // Downloaded
    if (origin.Downloaded) {
      if (origin.Downloaded.url.length > 15) {
        return `${origin.Downloaded.url.slice(0, 8)}...${origin.Downloaded.url.slice(origin.Downloaded.url.length - 8)}`;
      } else {
        return `Downloaded: ${origin.Downloaded.url}`;
      }
      // Unpacked
    } else if (origin.Unpacked) {
      if (origin.Unpacked.tool) {
        return `Unpacked: ${origin.Unpacked.tool}`;
      } else {
        return 'Unpacked';
      }
      // Transformed
    } else if (origin.Transformed) {
      if (origin.Transformed.tool) {
        return `Transformed: ${origin.Transformed.tool}`;
      } else {
        return 'Transformed';
      }
    } else if (origin.Incident) {
      return `Incident: ${origin.Incident.incident}`;
    } else if (origin.MemoryDump) {
      return `Memory Dump`;
      // leave label blank
    } else if (origin.Wire) {
      return `Wire: ${origin.Wire.sniffer}`;
    } else if (origin.Source) {
      return `Commit: ${origin.Source.commit.slice(0, 8)}`;
    }
  } else if (node.relationship.Association) {
    return `Association: ${node.relationship.Association.kind}`;
  } else if (node.relationship) {
    const tags = graph.data_map[target].Tag?.tags;
    if (tags === undefined) return '';
    const pairs = Object.keys(tags).map((key) => {
      return tags[key].map((value) => {
        return `${key}: ${value}`;
      });
    });
    //const nodeId: string = node.direction == Direction.To ? node.node: branchId;
    return `${pairs}`;
  }
  return '';
};
