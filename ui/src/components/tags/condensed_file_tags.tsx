import React, { Fragment } from 'react';
import { Col, Row } from 'react-bootstrap';

// project imports
import { filterIncludedTags, filterExcludedTags, FormattedFileInfoTagKeys, DangerTagKeys, MitreTagKeys } from './utilities';
import { TagBadge } from './tags';
import { Entities, Tags } from '@models';

interface CondensedFileTagProps {
  tags: Tags; // tags to display in condensed non-editable view
  excludeKeys: string[]; // tag keys to exclude when displaying tags
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
      <Row>
        <Col className="d-flex justify-content-center wrap">
          {Object.keys(generalTags)
            .sort()
            .map((tagKey) =>
              Object.keys(generalTags[tagKey])
                .sort()
                .map((tagValue) => (
                  <TagBadge
                    resource={Entities.File}
                    key={'General_' + tagValue}
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
                    key={'FileInfo_' + tagValue}
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
                    key={'Attack_' + tagValue}
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
                    key={'MBC_' + tagValue}
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
                    key={'FileInfo_' + tagValue}
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
                    key={'Results_' + tagValue}
                    tag={tagKey}
                    value={tagValue}
                    condensed={true}
                    action={'link'}
                  />
                )),
            )}
        </Col>
      </Row>
    </Fragment>
  );
};

export default CondensedFileTags;
