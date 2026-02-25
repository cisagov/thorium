// score the tags applied to an entity redo this later
export function scoreTags(tags: any): number {
  let score = 0;
  Object.keys(tags).map((tagKey) => {
    Object.keys(tags[tagKey]).map((tagValue) => {
      // check tag key and then add additional values based on certain tags
      if (['YaraRuleHit', 'ClamAV', 'AVHit'].includes(tagKey)) {
        score += 100;
      } else if (['MBC', 'ATT&CK'].includes(tagKey)) {
        score += 20;
      } else {
        score += 5;
      }
    });
  });
  return score;
}

export function scoreNode(node: any): number {
  if (node?.Sample) {
    if (node.Sample?.tags) {
      const tags = node.Sample.tags;
      return Math.min(350, Math.max(250, scoreTags(tags)));
    }
  } else if (node?.Repo) {
    return 400;
  } else if (node?.Tag) {
    return 350;
  } else if (node.Entity?.kind == 'Device') {
    return 450;
  } else if (node.Entity?.kind == 'FileSystem') {
    return 400;
  } else if (node.Entity?.kind == 'Vendor') {
    return 450;
  } else if (node.Entity?.kind == 'Collection') {
    return 450;
  } else if (node.Entity?.kind == 'Folder') {
    return 350;
  }
  return 300;
}

export function getNodeSize(score: number, numElements: number) {
  return score / 10;
}
