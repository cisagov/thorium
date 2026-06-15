import React, { Fragment } from 'react';
import styled from 'styled-components';

// project imports
import TagBadge from '../TagBadge';
import { DangerTagKeys, MitreTagKeys, FormattedFileInfoTagKeys } from '../tag_groups';
import { filterIncludedTags, filterExcludedTags } from '../utilities';
import { Tags } from '@models/tags';
import { Entities } from '@models/entities';

// spec: ../tags.spec.md

const TagContainer = styled.div`
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
`;

interface CondensedFileTagProps {
  tags: Tags;
  excludeKeys: string[];
}

const CondensedFileTags: React.FC<CondensedFileTagProps> = ({ tags, excludeKeys }) => {
  const generalExcludeTags = [...FormattedFileInfoTagKeys, ...excludeKeys, ...MitreTagKeys, ...DangerTagKeys, 'Results'];
  const dangerTags = filterIncludedTags(tags, DangerTagKeys);
  const generalTags = filterExcludedTags(tags, generalExcludeTags);
  const fileInfoTags = filterIncludedTags(tags, FormattedFileInfoTagKeys);
  const attackTags = filterIncludedTags(tags, ['ATT&CK']);
  const mbcTags = filterIncludedTags(tags, ['MBC']);
  const resultsTags = filterExcludedTags(filterIncludedTags(tags, ['Results']), excludeKeys);
  // total number of displayed tags (not including those filtered out)
  const tagCount =
    Object.keys(dangerTags).length +
    Object.keys(generalTags).length +
    Object.keys(fileInfoTags).length +
    Object.keys(attackTags).length +
    Object.keys(mbcTags).length +
    Object.keys(resultsTags).length;
  return (
    <Fragment>
      {tagCount > 0 && <hr />}
      <TagContainer>
        {Object.keys(generalTags)
          .sort()
          .map((tagKey) =>
            Object.keys(generalTags[tagKey])
              .sort()
              .map((tagValue) => (
                <TagBadge
                  resource={Entities.File}
                  key={'General_' + tagKey + '_' + tagValue}
                  tag={tagKey}
                  value={tagValue}
                  condensed={true}
                  action={'link'}
                />
              )),
          )}
        {Object.keys(dangerTags)
          .sort()
          .map((tagKey) =>
            Object.keys(dangerTags[tagKey])
              .sort()
              .map((tagValue) => (
                <TagBadge
                  resource={Entities.File}
                  key={'Danger_' + tagKey + '_' + tagValue}
                  tag={tagKey}
                  value={tagValue}
                  condensed={true}
                  action={'link'}
                />
              )),
          )}
        {Object.keys(attackTags)
          .sort()
          .map((tagKey) =>
            Object.keys(attackTags[tagKey])
              .sort()
              .map((tagValue) => (
                <TagBadge
                  resource={Entities.File}
                  key={'Attack_' + tagKey + '_' + tagValue}
                  tag={'ATT&CK'}
                  value={tagValue}
                  condensed={true}
                  action={'link'}
                />
              )),
          )}
        {Object.keys(mbcTags)
          .sort()
          .map((tagKey) =>
            Object.keys(mbcTags[tagKey])
              .sort()
              .map((tagValue) => (
                <TagBadge
                  resource={Entities.File}
                  key={'MBC_' + tagKey + '_' + tagValue}
                  tag={'MBC'}
                  value={tagValue}
                  condensed={true}
                  action={'link'}
                />
              )),
          )}
        {Object.keys(fileInfoTags)
          .sort()
          .map((tagKey) =>
            Object.keys(fileInfoTags[tagKey])
              .sort()
              .map((tagValue) => (
                <TagBadge
                  resource={Entities.File}
                  key={'FileInfo_' + tagKey + '_' + tagValue}
                  tag={tagKey}
                  value={tagValue}
                  condensed={true}
                  action={'link'}
                />
              )),
          )}
        {Object.keys(resultsTags)
          .sort()
          .map((tagKey) =>
            Object.keys(resultsTags[tagKey])
              .sort()
              .map((tagValue) => (
                <TagBadge
                  resource={Entities.File}
                  key={'Results_' + tagKey + '_' + tagValue}
                  tag={tagKey}
                  value={tagValue}
                  condensed={true}
                  action={'link'}
                />
              )),
          )}
      </TagContainer>
    </Fragment>
  );
};

export default CondensedFileTags;
