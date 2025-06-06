// score the tags applied to an entity redo this later
export function scoreTags(tags: any): number {
  let score = 0;
  Object.keys(tags).map((tagKey) => {
    Object.keys(tags[tagKey]).map((tagValue) => {
      // bcheck tag key and then add additional values based on certain tags
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
      console.log(scoreTags(tags));
      return Math.min(350, Math.max(250, scoreTags(tags)));
    }
  } else if (node?.Repo) {
    return 400;
  } else if (node?.Tag) {
    // tags are specified by the user, they are important
    return 400;
  } else if (node.Entity?.kind == 'Device') {
    // tags are specified by the user, they are important
    return 450;
  } else if (node.Entity?.kind == 'Vendor') {
    // tags are specified by the user, they are important
    return 450;
  }
  return 0;
}

export function getNodeSize(score: number, numElements: number) {
  return score / 10;
}
